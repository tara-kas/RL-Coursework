"""
Hybrid CNN-Transformer AlphaZero self-play training for 9x9 Gomoku.
Run: python train_hybrid.py [--options]
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch

from train import evaluate_alphazero, progress_bar, self_play
from src.Bots.alpha_zero_hybrid import AlphaZeroHybrid, Bot
from src.gomoku_utils import preprocess_board
from src.model_loader import load_weights, save_weights as save_model_weights


# =========================
# Default Hyperparameters
# =========================
DEFAULT_BOARD_SIZE = 9
DEFAULT_NUM_SIMULATIONS = 800
DEFAULT_GAMES_PER_ITERATION = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR_MAX = 1e-3
DEFAULT_LR_MIN = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_ITERATIONS = 100
DEFAULT_TRAIN_EPOCHS = 2
DEFAULT_VALUE_COEF = 1.0
DEFAULT_GRAD_CLIP_NORM = 1.0


def _augment_symmetries(
    states: np.ndarray, policies: np.ndarray, board_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    8-fold augmentation: 4 rotations x (no flip, horizontal flip).
    states shape: (B, 3, 9, 9), policies shape: (B, 81)
    """
    out_states: list[np.ndarray] = []
    out_policies: list[np.ndarray] = []
    for s, p in zip(states, policies, strict=False):
        p2 = p.reshape(board_size, board_size)
        for k in range(4):
            s_rot = np.rot90(s, k=k, axes=(1, 2)).copy()
            p_rot = np.rot90(p2, k=k).copy()
            out_states.append(s_rot)
            out_policies.append(p_rot.reshape(-1))

            s_flip = np.flip(s_rot, axis=2).copy()
            p_flip = np.flip(p_rot, axis=1).copy()
            out_states.append(s_flip)
            out_policies.append(p_flip.reshape(-1))
    return np.stack(out_states), np.stack(out_policies)


def _policy_cross_entropy(policy_pred: torch.Tensor, policy_target: torch.Tensor) -> torch.Tensor:
    # Cross-entropy with soft targets from MCTS visit distribution.
    return -(policy_target * torch.log(policy_pred + 1e-9)).sum(dim=1).mean()


def train_step(
    batch: list[tuple[np.ndarray, int, np.ndarray, float]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    value_coef: float,
    grad_clip_norm: float,
    board_size: int,
) -> tuple[float, float]:
    model.train()

    states_list: list[np.ndarray] = []
    policy_targets_list: list[np.ndarray] = []
    value_targets_list: list[float] = []
    masks_list: list[np.ndarray] = []

    for board, current_player, policy, z in batch:
        # Build requested input planes: current stones, opponent stones, color-to-play.
        base = preprocess_board(board, current_player)
        color_plane = np.ones_like(base[0], dtype=np.float32)
        planes = np.stack([base[0], base[1], color_plane], axis=0)
        states_list.append(planes)
        policy_targets_list.append(policy)
        value_targets_list.append(z)
        legal_mask = (board == -1).astype(np.float32).flatten()
        masks_list.append(legal_mask)

    states_np = np.stack(states_list).astype(np.float32)
    policy_np = np.stack(policy_targets_list).astype(np.float32)
    value_np = np.array(value_targets_list, dtype=np.float32)
    masks_np = np.stack(masks_list).astype(np.float32)

    # Data augmentation over the batch.
    states_np, policy_np = _augment_symmetries(states_np, policy_np, board_size)
    value_np = np.repeat(value_np, 8)
    masks_np = np.repeat(masks_np, 8, axis=0)

    states = torch.tensor(states_np, dtype=torch.float32, device=device)
    policy_targets = torch.tensor(policy_np, dtype=torch.float32, device=device)
    value_targets = torch.tensor(value_np, dtype=torch.float32, device=device).unsqueeze(1)
    masks = torch.tensor(masks_np, dtype=torch.float32, device=device)

    policy_pred, value_pred = model(states, masks)
    policy_loss = _policy_cross_entropy(policy_pred, policy_targets)
    value_loss = torch.nn.functional.mse_loss(value_pred, value_targets)
    loss = policy_loss + value_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optimizer.step()
    scheduler.step()

    return float(policy_loss.item()), float(value_loss.item())


def save_attention_maps(
    model: AlphaZeroHybrid,
    board: np.ndarray,
    current_player: int,
    output_path: str,
    device: torch.device,
) -> str:
    """
    Save average attention map image for one board state.
    Returns output path; requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for attention visualization") from exc

    model.eval()
    base = preprocess_board(board, current_player)
    planes = np.stack([base[0], base[1], np.ones_like(base[0], dtype=np.float32)], axis=0)
    x = torch.tensor(planes[None, ...], dtype=torch.float32, device=device)
    legal_mask = torch.tensor((board == -1).astype(np.float32).reshape(1, -1), device=device)

    with torch.inference_mode():
        _, _, attn_maps = model(x, legal_mask, return_attention=True)

    if not attn_maps:
        raise RuntimeError("No attention maps produced by model.")

    # Last block, average over heads and query tokens -> key-token importance.
    attn = attn_maps[-1][0].mean(dim=0).mean(dim=0).reshape(9, 9).cpu().numpy()
    plt.figure(figsize=(5, 4))
    plt.imshow(attn, cmap="viridis")
    plt.colorbar()
    plt.title("Hybrid Transformer Attention (last layer)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid CNN-Transformer AlphaZero training for 9x9 Gomoku")
    parser.add_argument("--board_size", type=int, default=DEFAULT_BOARD_SIZE, choices=[9])
    parser.add_argument("--num_simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--games_per_iteration", type=int, default=DEFAULT_GAMES_PER_ITERATION)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr_max", type=float, default=DEFAULT_LR_MAX)
    parser.add_argument("--lr_min", type=float, default=DEFAULT_LR_MIN)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--train_epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--checkpoint_dir", type=str, default="weights")
    parser.add_argument("--save_best_path", type=str, default="weights/hybrid_best.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--value_coef", type=float, default=DEFAULT_VALUE_COEF)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--mcts_batch_size", type=int, default=32)
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument("--self_play_temp", type=float, default=1.0)
    parser.add_argument("--temp_moves", type=int, default=30)
    parser.add_argument("--league_prob", type=float, default=0.25)
    parser.add_argument("--heuristic_prob", type=float, default=0.2)
    parser.add_argument("--league_pool_size", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_root_noise", action="store_true")
    parser.add_argument("--root_dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--root_dirichlet_epsilon", type=float, default=0.25)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--eval_games", type=int, default=50)
    parser.add_argument("--best_by", type=str, default="heuristic", choices=("loss", "heuristic"))
    parser.add_argument("--eval_games_best", type=int, default=100)
    parser.add_argument("--grad_clip_norm", type=float, default=DEFAULT_GRAD_CLIP_NORM)
    parser.add_argument("--save_attention_example", type=str, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        print(f"Using device: {device} (CUDA not available)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model = AlphaZeroHybrid(board_size=args.board_size).to(device)
    iteration_offset = 0
    if args.resume and os.path.isfile(args.resume):
        load_weights(model, args.resume, device)
        print(f"Resumed from {args.resume}")
        base = os.path.basename(args.resume)
        if base.startswith("hybrid_checkpoint_") and base.endswith(".pt"):
            try:
                iteration_offset = int(base[len("hybrid_checkpoint_") : -len(".pt")])
            except ValueError:
                pass

    bot = Bot(
        model=model,
        board_size=args.board_size,
        device=device,
        num_simulations=args.num_simulations,
        compile_model=not args.no_compile,
        mcts_batch_size=args.mcts_batch_size,
        use_amp=args.amp,
    )

    optimizer = torch.optim.AdamW(
        bot.model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay
    )

    # Step-wise cosine decay from lr_max -> lr_min.
    est_batches_per_epoch = max(1, math.ceil((args.games_per_iteration * 45) / args.batch_size))
    total_scheduler_steps = max(1, args.iterations * args.train_epochs * est_batches_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_scheduler_steps, eta_min=args.lr_min
    )

    buffer: list[tuple[np.ndarray, int, np.ndarray, float]] = []
    buffer_max_size = args.games_per_iteration * 200
    best_loss = float("inf")
    best_heuristic_wr = -1.0
    checkpoint_pool: list[str] = []
    total_display = iteration_offset + args.iterations

    def log_self_play(current: int, total: int) -> None:
        msg = progress_bar(current, total, width=25, prefix="  Self-play ", suffix=" games")
        sys.stdout.write(f"\r{msg}")
        sys.stdout.flush()

    for i in range(args.iterations):
        iteration = iteration_offset + i + 1
        print(progress_bar(iteration, total_display, width=30, prefix="Iteration ", suffix=""))

        league_bot = None
        if checkpoint_pool and args.league_prob > 0:
            league_path = random.choice(checkpoint_pool)
            if os.path.isfile(league_path):
                league_model = AlphaZeroHybrid(board_size=args.board_size).to(device)
                load_weights(league_model, league_path, device)
                league_bot = Bot(
                    model=league_model,
                    board_size=args.board_size,
                    device=device,
                    num_simulations=args.num_simulations,
                    compile_model=not args.no_compile,
                    mcts_batch_size=args.mcts_batch_size,
                    use_amp=args.amp,
                )
                league_bot.model.eval()

        new_data, self_play_stats = self_play(
            bot,
            args.board_size,
            args.games_per_iteration,
            c_puct=args.c_puct,
            self_play_temp=args.self_play_temp,
            temp_moves=args.temp_moves,
            progress_callback=log_self_play,
            league_bot=league_bot,
            league_prob=args.league_prob,
            heuristic_prob=args.heuristic_prob,
            add_root_noise=not args.no_root_noise,
            root_dirichlet_alpha=args.root_dirichlet_alpha,
            root_dirichlet_epsilon=args.root_dirichlet_epsilon,
        )
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        buffer.extend(new_data)
        if len(buffer) > buffer_max_size:
            buffer = buffer[-buffer_max_size:]

        random.shuffle(buffer)
        total_pl, total_vl, n_batches = 0.0, 0.0, 0
        for _epoch in range(args.train_epochs):
            for start in range(0, len(buffer), args.batch_size):
                batch = buffer[start : start + args.batch_size]
                if len(batch) < 2:
                    continue
                pl, vl = train_step(
                    batch=batch,
                    model=bot.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    value_coef=args.value_coef,
                    grad_clip_norm=args.grad_clip_norm,
                    board_size=args.board_size,
                )
                total_pl += pl
                total_vl += vl
                n_batches += 1

        total_games = self_play_stats["wins"] + self_play_stats["losses"] + self_play_stats["draws"]
        game_success_rate = self_play_stats["wins"] / total_games if total_games else 0.0
        game_type_log = (
            f"league={self_play_stats['games_league']} "
            f"heuristic={self_play_stats['games_heuristic']} "
            f"self_play={self_play_stats['games_self_play']}"
        )

        if n_batches > 0:
            avg_pl = total_pl / n_batches
            avg_vl = total_vl / n_batches
            avg_loss = avg_pl + args.value_coef * avg_vl
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Loss: policy={avg_pl:.4f} value={avg_vl:.4f} total={avg_loss:.4f} "
                f"buffer={len(buffer)} lr={current_lr:.6f} game_success_rate={game_success_rate:.2f} [{game_type_log}]"
            )
            if args.best_by == "loss" and avg_loss < best_loss:
                best_loss = avg_loss
                save_model_weights(bot.model, args.save_best_path)
                print(f"  -> saved best (by loss) to {args.save_best_path}")
        else:
            print(
                f"  Buffer size {len(buffer)} too small; skipping train "
                f"game_success_rate={game_success_rate:.2f} [{game_type_log}]"
            )

        if iteration % args.eval_freq == 0 or iteration == total_display:
            eval_random = evaluate_alphazero(
                bot=bot,
                board_size=args.board_size,
                num_games=args.eval_games,
                opponent="random",
            )
            n_heuristic = args.eval_games_best if args.best_by == "heuristic" else args.eval_games
            eval_heuristic = evaluate_alphazero(
                bot=bot,
                board_size=args.board_size,
                num_games=n_heuristic,
                opponent="heuristic",
            )
            print(
                f"  Eval: vs_random win_rate={eval_random['win_rate']:.2f} "
                f"vs_heuristic win_rate={eval_heuristic['win_rate']:.2f}"
            )
            if args.best_by == "heuristic" and eval_heuristic["win_rate"] > best_heuristic_wr:
                best_heuristic_wr = eval_heuristic["win_rate"]
                save_model_weights(bot.model, args.save_best_path)
                print(f"  -> saved best (by heuristic) to {args.save_best_path}")

        if iteration % 5 == 0 or iteration == total_display:
            ckpt_path = os.path.join(args.checkpoint_dir, f"hybrid_checkpoint_{iteration}.pt")
            save_model_weights(bot.model, ckpt_path)
            checkpoint_pool.append(ckpt_path)
            checkpoint_pool = checkpoint_pool[-args.league_pool_size:]
            print(f"  -> checkpoint {ckpt_path}")

    if args.save_attention_example is not None and buffer:
        board, current_player, _, _ = random.choice(buffer)
        saved_path = save_attention_maps(
            model=model,
            board=board,
            current_player=current_player,
            output_path=args.save_attention_example,
            device=device,
        )
        print(f"Saved attention map to {saved_path}")

    print(progress_bar(total_display, total_display, width=30, prefix="Done.       ", suffix=""))
    print("Hybrid AlphaZero training complete.")


if __name__ == "__main__":
    main()
