"""
Standalone AlphaZero self-play training for Gomoku. No Pygame/UI dependencies.
Run: python train.py [--options]
"""
import argparse
import math
import multiprocessing
import os
import random
import sys
from typing import Callable

import numpy as np
import torch

from src.gomoku_game import (
    GAME_NOT_OVER,
    WIN,
    DRAW,
    apply_move,
    get_game_result,
    get_legal_moves,
)
from src.gomoku_utils import preprocess_board
from src.Bots.alpha_zero_resnet import (
    AlphaZeroTransform as AlphaZeroResNet,
    Bot as AlphaZeroResNetBot,
)
from src.Bots.alpha_zero_transformer import (
    AlphaZeroTransform as AlphaZeroTransformer,
    Bot as AlphaZeroTransformerBot,
)
from src.Bots.alpha_zero_hybrid import AlphaZeroHybrid, Bot as HybridBot
from src.Bots.heuristic_tactical import predict as heuristic_predict
from src.Bots.dqn import (
    DQN,
    ReplayBuffer,
    get_epsilon,
    dqn_self_play,
    dqn_train_step,
    evaluate_dqn,
)
from src.model_loader import save_weights as save_model_weights, load_weights


def _get_alphazero_impl(
    agent_type: str,
) -> tuple[type[torch.nn.Module], type[AlphaZeroResNetBot | AlphaZeroTransformerBot], str]:
    """Return (model_class, bot_class, variant_label) for AlphaZero variants."""
    if agent_type in ("alphazero", "alphazero-resnet"):
        return AlphaZeroResNet, AlphaZeroResNetBot, "resnet"
    if agent_type == "alphazero-transformer":
        return AlphaZeroTransformer, AlphaZeroTransformerBot, "transformer"
    raise ValueError(f"Unsupported AlphaZero agent_type: {agent_type}")


def progress_bar(
    current: int,
    total: int,
    width: int = 30,
    prefix: str = "",
    suffix: str = "",
    fill: str = "=",
    head: str = ">",
    empty: str = " ",
) -> str:
    """Build a single-line progress bar string: [=====>    ] 50% (current/total)."""
    if total <= 0:
        pct = 100.0
        filled = width
    else:
        pct = 100.0 * current / total
        filled = int(width * current / total)
    bar = fill * max(0, filled - 1) + (head if filled > 0 else "") + empty * max(0, width - filled)
    return f"{prefix}[{bar}] {pct:5.1f}% ({current}/{total}){suffix}"


def self_play(
    bot: AlphaZeroResNetBot | AlphaZeroTransformerBot | HybridBot,
    board_size: int,
    num_games: int,
    c_puct: float = 1.5,
    self_play_temp: float = 1.0,
    temp_moves: int = 30,
    progress_callback: Callable[[int, int], None] | None = None,
    league_bot: AlphaZeroResNetBot | AlphaZeroTransformerBot | HybridBot | None = None,
    league_prob: float = 0.0,
    heuristic_prob: float = 0.0,
    add_root_noise: bool = True,
    root_dirichlet_alpha: float = 0.3,
    root_dirichlet_epsilon: float = 0.25,
) -> tuple[list[tuple[np.ndarray, int, np.ndarray, float]], dict[str, int]]:
    """
    Run num_games. Each game is self-play, league (vs league_bot), or heuristic (vs tactical bot)
    with probabilities (1 - league_prob - heuristic_prob), league_prob, heuristic_prob.
    Returns:
      - list of (board, current_player, policy, z)
      - stats dict with wins/losses/draws and game type counts for player 0
    """
    bot.model.eval()
    buffer: list[tuple[np.ndarray, int, np.ndarray, float]] = []
    stats = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "games_league": 0,
        "games_heuristic": 0,
        "games_self_play": 0,
    }

    for g in range(num_games):
        if progress_callback is not None:
            progress_callback(g + 1, num_games)
        r = random.random()
        if league_bot is not None and r < league_prob:
            opponent = league_bot
            stats["games_league"] += 1
        elif r < league_prob + heuristic_prob:
            opponent = heuristic_predict
            stats["games_heuristic"] += 1
        else:
            opponent = None
            stats["games_self_play"] += 1

        board = np.full((board_size, board_size), -1, dtype=np.int32)
        current_player = 0
        game_history: list[tuple[np.ndarray, int, np.ndarray]] = []

        while True:
            if opponent is None or current_player == 0:
                temp = self_play_temp if len(game_history) < temp_moves else 0.0
                move, policy = bot.get_move_and_policy(
                    board.copy(),
                    current_player,
                    c_puct=c_puct,
                    temperature=temp,
                    add_root_noise=add_root_noise,
                    dirichlet_alpha=root_dirichlet_alpha,
                    dirichlet_epsilon=root_dirichlet_epsilon,
                )
                if opponent is None:
                    game_history.append((board.copy(), current_player, policy))
                else:
                    game_history.append((board.copy(), 0, policy))
            else:
                if hasattr(opponent, "predict"):
                    move = opponent.predict(board.copy(), current_player)
                else:
                    move = opponent(board.copy(), current_player)

            board = apply_move(board, move, current_player)
            result = get_game_result(board, move, current_player)

            if result != GAME_NOT_OVER:
                if result == WIN:
                    winner = current_player
                else:
                    winner = -1
                if winner == 0:
                    stats["wins"] += 1
                elif winner == 1:
                    stats["losses"] += 1
                else:
                    stats["draws"] += 1
                if opponent is not None:
                    z = 1.0 if winner == 0 else (-1.0 if winner == 1 else 0.0)
                    for b, _p, pi in game_history:
                        buffer.append((b, 0, pi, z))
                else:
                    for b, p, pi in game_history:
                        z = 0.0 if winner == -1 else (1.0 if p == winner else -1.0)
                        buffer.append((b, p, pi, z))
                break
            current_player = 1 - current_player

    return buffer, stats


def _worker_self_play(
    worker_id: int,
    state_dict: dict,
    num_games: int,
    board_size: int,
    num_simulations: int,
    alphazero_agent_type: str,
    mcts_batch_size: int,
    c_puct: float,
    self_play_temp: float,
    temp_moves: int,
    league_prob: float,
    heuristic_prob: float,
    league_path: str | None,
    device_str: str,
    seed_base: int | None,
    use_amp: bool,
    compile_model: bool,
    add_root_noise: bool,
    root_dirichlet_alpha: float,
    root_dirichlet_epsilon: float,
) -> tuple[list[tuple[np.ndarray, int, np.ndarray, float]], dict[str, int]]:
    """
    Worker for parallel self-play. Loads model from state_dict, runs num_games, returns buffer.
    Must be top-level for pickling in multiprocessing.
    """
    if seed_base is not None:
        random.seed(seed_base + worker_id)
        np.random.seed(seed_base + worker_id)
        torch.manual_seed(seed_base + worker_id)

    device = torch.device(device_str)
    model_cls, bot_cls, _ = _get_alphazero_impl(alphazero_agent_type)
    model = model_cls(board_size=board_size).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    bot = bot_cls(
        model=model,
        board_size=board_size,
        device=device,
        num_simulations=num_simulations,
        compile_model=compile_model,
        mcts_batch_size=mcts_batch_size,
        use_amp=use_amp,
    )

    league_bot: AlphaZeroResNetBot | AlphaZeroTransformerBot | None = None
    if league_path and league_prob > 0 and os.path.isfile(league_path):
        league_model = model_cls(board_size=board_size).to(device)
        load_weights(league_model, league_path, device)
        league_bot = bot_cls(
            model=league_model,
            board_size=board_size,
            device=device,
            num_simulations=num_simulations,
            compile_model=compile_model,
            mcts_batch_size=mcts_batch_size,
            use_amp=use_amp,
        )
        league_bot.model.eval()

    return self_play(
        bot,
        board_size,
        num_games,
        c_puct=c_puct,
        self_play_temp=self_play_temp,
        temp_moves=temp_moves,
        progress_callback=None,
        league_bot=league_bot,
        league_prob=league_prob,
        heuristic_prob=heuristic_prob,
        add_root_noise=add_root_noise,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_dirichlet_epsilon=root_dirichlet_epsilon,
    )


def train_step(
    batch: list[tuple[np.ndarray, int, np.ndarray, float]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    board_size: int,
    device: torch.device,
    value_coef: float = 1.0,
) -> tuple[float, float]:
    """One training step on a batch. Returns (policy_loss, value_loss)."""
    model.train()

    states_list = []
    policy_targets_list = []
    value_targets_list = []
    masks_list = []

    for board, current_player, policy, z in batch:
        planes = preprocess_board(board, current_player)
        states_list.append(planes)
        policy_targets_list.append(policy)
        value_targets_list.append(z)
        legal_mask = (board == -1).astype(np.float32).flatten()
        masks_list.append(legal_mask)

    states = torch.tensor(np.stack(states_list), dtype=torch.float32, device=device)
    policy_targets = torch.tensor(
        np.stack(policy_targets_list), dtype=torch.float32, device=device
    )
    value_targets = torch.tensor(
        np.stack(value_targets_list), dtype=torch.float32, device=device
    ).unsqueeze(1)
    masks = torch.tensor(np.stack(masks_list), dtype=torch.float32, device=device)

    policy_pred, value_pred = model(states, masks)

    # Policy: soft targets from MCTS visit distribution -> KL divergence
    policy_loss = torch.nn.functional.kl_div(
        torch.log(policy_pred + 1e-9),
        policy_targets,
        reduction="batchmean",
    )
    value_loss = torch.nn.functional.mse_loss(value_pred, value_targets)

    loss = policy_loss + value_coef * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def _augment_symmetries(
    states: np.ndarray,
    policies: np.ndarray,
    board_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """8-fold augmentation: 4 rotations x (no flip, horizontal flip)."""
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


def train_step_hybrid(
    batch: list[tuple[np.ndarray, int, np.ndarray, float]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    board_size: int,
    device: torch.device,
    value_coef: float = 1.0,
    grad_clip_norm: float = 1.0,
) -> tuple[float, float]:
    """One hybrid training step with 8-fold augmentation and grad clipping."""
    model.train()

    states_list = []
    policy_targets_list = []
    value_targets_list = []
    masks_list = []

    for board, current_player, policy, z in batch:
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

    states_np, policy_np = _augment_symmetries(states_np, policy_np, board_size)
    value_np = np.repeat(value_np, 8)
    masks_np = np.repeat(masks_np, 8, axis=0)

    states = torch.tensor(states_np, dtype=torch.float32, device=device)
    policy_targets = torch.tensor(policy_np, dtype=torch.float32, device=device)
    value_targets = torch.tensor(value_np, dtype=torch.float32, device=device).unsqueeze(1)
    masks = torch.tensor(masks_np, dtype=torch.float32, device=device)

    policy_pred, value_pred = model(states, masks)
    policy_loss = -(policy_targets * torch.log(policy_pred + 1e-9)).sum(dim=1).mean()
    value_loss = torch.nn.functional.mse_loss(value_pred, value_targets)

    loss = policy_loss + value_coef * value_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optimizer.step()
    scheduler.step()

    return policy_loss.item(), value_loss.item()


def evaluate_alphazero(
    bot: AlphaZeroResNetBot | AlphaZeroTransformerBot | HybridBot,
    board_size: int,
    num_games: int,
    opponent: str,
) -> dict[str, float]:
    """
    Evaluate AlphaZero bot as player 0 vs random or heuristic opponent.
    Returns wins/losses/draws/win_rate/avg_return.
    """
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games):
        board = np.full((board_size, board_size), -1, dtype=np.int32)
        current_player = 0
        while True:
            if current_player == 0:
                move = bot.predict(board.copy(), current_player=current_player)
            else:
                legal = get_legal_moves(board)
                if not legal:
                    break
                if opponent == "random":
                    move = random.choice(legal)
                elif opponent == "heuristic":
                    move = heuristic_predict(board.copy(), current_player)
                else:
                    raise ValueError(f"Unsupported AlphaZero eval opponent: {opponent}")

            board = apply_move(board, move, current_player)
            result = get_game_result(board, move, current_player)
            if result != GAME_NOT_OVER:
                if result == WIN:
                    winner = current_player
                else:
                    winner = -1
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1
                break
            current_player = 1 - current_player

    total = wins + losses + draws
    win_rate = wins / total if total else 0.0
    avg_return = (wins - losses) / total if total else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "avg_return": avg_return,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gomoku self-play training (AlphaZero, Hybrid, or DQN)")
    parser.add_argument(
        "--agent_type",
        type=str,
        default="alphazero-resnet",
        choices=[
            "alphazero",
            "alphazero-resnet",
            "alphazero-transformer",
            "hybrid",
            "alphazero-hybrid",
            "dqn",
        ],
        help="Agent type: alphazero-resnet, alphazero-transformer, hybrid (or alphazero-hybrid), or dqn",
    )
    parser.add_argument(
        "--board_size",
        type=int,
        default=15,
        choices=[9, 15],
        help="Board size (must be 9 or 15)",
    )
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--games_per_iteration", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate; DQN may benefit from higher (e.g. 3e-4) if loss is stuck near 0")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="weights")
    parser.add_argument("--save_best_path", type=str, default="weights/best.pt")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--value_coef", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile for the model")
    parser.add_argument("--mcts_batch_size", type=int, default=32, help="Batch size for MCTS leaf evaluation")
    parser.add_argument("--c_puct", type=float, default=1.5, help="MCTS exploration constant")
    parser.add_argument("--self_play_temp", type=float, default=1.0, help="Temperature for move sampling in self-play (first temp_moves)")
    parser.add_argument("--temp_moves", type=int, default=30, help="Number of moves per game with temperature; after that argmax")
    parser.add_argument("--league_prob", type=float, default=0.25, help="Probability of playing vs a past checkpoint")
    parser.add_argument("--heuristic_prob", type=float, default=0.2, help="Probability of playing vs heuristic tactical bot")
    parser.add_argument("--league_pool_size", type=int, default=5, help="Max past checkpoints to keep in league pool")
    parser.add_argument("--amp", action="store_true", help="Use FP16 autocast in MCTS")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel self-play workers (1 = no parallelism)")
    parser.add_argument("--worker_device", type=str, default="cpu", help="Device for parallel workers (main process keeps GPU for training)")
    parser.add_argument("--no_root_noise", action="store_true", help="Disable Dirichlet root noise during AlphaZero self-play")
    parser.add_argument("--root_dirichlet_alpha", type=float, default=0.3, help="AlphaZero root Dirichlet concentration for self-play")
    parser.add_argument("--root_dirichlet_epsilon", type=float, default=0.25, help="AlphaZero root prior mixing weight for Dirichlet noise")
    parser.add_argument("--az_eval_freq", type=int, default=10, help="AlphaZero evaluation every N iterations (win rate vs random/heuristic)")
    parser.add_argument("--az_best_by", type=str, default="heuristic", choices=("loss", "heuristic"), help="AlphaZero: save best by loss or by heuristic win rate")
    parser.add_argument("--az_eval_games_best", type=int, default=100, help="AlphaZero eval games vs heuristic when saving best by heuristic")
    # Hybrid-specific (ignored unless agent_type == hybrid)
    parser.add_argument("--hybrid_lr_min", type=float, default=1e-4, help="Hybrid cosine scheduler minimum LR")
    parser.add_argument("--hybrid_weight_decay", type=float, default=1e-4, help="Hybrid AdamW weight decay")
    parser.add_argument("--hybrid_grad_clip_norm", type=float, default=1.0, help="Hybrid gradient clipping max_norm")
    # DQN-specific (ignored unless agent_type == dqn)
    parser.add_argument("--gamma", type=float, default=0.99, help="DQN discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="DQN initial exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="DQN final exploration")
    parser.add_argument("--epsilon_decay_steps", type=int, default=150000, help="DQN epsilon decay steps (longer exploration)")
    parser.add_argument("--replay_buffer_size", type=int, default=100000, help="DQN replay buffer capacity")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="DQN target network sync interval (steps)")
    parser.add_argument("--train_steps_per_iteration", type=int, default=500, help="DQN training steps per iteration")
    parser.add_argument("--dqn_terminal_fraction", type=float, default=0.5, help="DQN fraction of batch from terminal transitions (stronger reward signal)")
    parser.add_argument("--eval_freq", type=int, default=10, help="DQN evaluation every N iterations (win rate vs random/heuristic)")
    parser.add_argument("--eval_games", type=int, default=50, help="Games per DQN eval vs random and vs heuristic")
    parser.add_argument("--eval_games_best", type=int, default=100, help="Games for heuristic eval when saving best by heuristic (lower variance)")
    parser.add_argument("--best_by", type=str, default="heuristic", choices=("loss", "heuristic"), help="DQN: save best by loss or by heuristic win rate")
    parser.add_argument("--heuristic_win_bonus", type=float, default=0.0, help="DQN: extra reward for winning vs heuristic (e.g. 0.1-0.3); 0 = off")
    parser.add_argument("--heuristic_prob_start", type=float, default=None, help="DQN: start heuristic_prob at this value and decay to heuristic_prob (curriculum)")
    parser.add_argument("--heuristic_prob_decay_iters", type=int, default=None, help="DQN: iterations over which to decay heuristic_prob_start to heuristic_prob")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        print(f"Using device: {device} (CUDA not available)")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.agent_type in ("alphazero", "alphazero-resnet", "alphazero-transformer"):
        _run_alphazero_training(args, device)
    elif args.agent_type in ("hybrid", "alphazero-hybrid"):
        _run_hybrid_training(args, device)
    elif args.agent_type == "dqn":
        _run_dqn_training(args, device)
    else:
        raise ValueError(f"Unknown agent_type: {args.agent_type}")


def _run_alphazero_training(args: argparse.Namespace, device: torch.device) -> None:
    """AlphaZero self-play and training loop."""
    model_cls, bot_cls, variant_label = _get_alphazero_impl(args.agent_type)
    print(f"AlphaZero variant: {variant_label}")
    model = model_cls(board_size=args.board_size).to(device)
    iteration_offset = 0
    if args.resume and os.path.isfile(args.resume):
        load_weights(model, args.resume, device)
        print(f"Resumed from {args.resume}")
        # Continue checkpoint numbering from the resumed file (e.g. checkpoint_100.pt -> next is 101)
        base = os.path.basename(args.resume)
        if base.startswith("checkpoint_") and base.endswith(".pt"):
            try:
                iteration_offset = int(base[len("checkpoint_") : -len(".pt")])
            except ValueError:
                pass

    bot = bot_cls(
        model=model,
        board_size=args.board_size,
        device=device,
        num_simulations=args.num_simulations,
        compile_model=not args.no_compile,
        mcts_batch_size=args.mcts_batch_size,
        use_amp=args.amp,
    )
    optimizer = torch.optim.Adam(bot.model.parameters(), lr=args.learning_rate)

    buffer: list[tuple[np.ndarray, int, np.ndarray, float]] = []
    buffer_max_size = args.games_per_iteration * 200  # rough upper bound positions per game
    best_loss = float("inf")
    best_heuristic_wr = -1.0
    total_iterations = args.iterations
    checkpoint_pool: list[str] = []

    def log_self_play(current: int, total: int) -> None:
        msg = progress_bar(current, total, width=25, prefix="  Self-play ", suffix=" games")
        sys.stdout.write(f"\r{msg}")
        sys.stdout.flush()

    def log_train(epoch: int, total_epochs: int, batch: int, total_batches: int) -> None:
        step = (epoch * total_batches + batch) / max(1, total_epochs * total_batches)
        filled = int(25 * step)
        bar = "=" * max(0, filled - 1) + (">" if filled > 0 else "") + " " * max(0, 25 - filled)
        pct = 100.0 * step
        sys.stdout.write(
            f"\r  Train     [{bar}] {pct:5.1f}% (epoch {epoch + 1}/{total_epochs}, batch {batch + 1}/{total_batches})"
        )
        sys.stdout.flush()

    total_display = iteration_offset + total_iterations
    for i in range(total_iterations):
        iteration = iteration_offset + i + 1
        # Overall iteration progress (e.g. "Iteration 101/600")
        iter_msg = progress_bar(iteration, total_display, width=30, prefix="Iteration ", suffix="")
        print(iter_msg)

        # Select league path for this iteration (if any)
        league_path: str | None = None
        league_bot: AlphaZeroResNetBot | AlphaZeroTransformerBot | None = None
        if checkpoint_pool and args.league_prob > 0:
            league_path = random.choice(checkpoint_pool)
            if args.num_workers == 1:
                league_model = model_cls(board_size=args.board_size).to(device)
                load_weights(league_model, league_path, device)
                league_bot = bot_cls(
                    model=league_model,
                    board_size=args.board_size,
                    device=device,
                    num_simulations=args.num_simulations,
                    compile_model=not args.no_compile,
                    mcts_batch_size=args.mcts_batch_size,
                    use_amp=args.amp,
                )
                league_bot.model.eval()

        # Self-play: parallel workers or single process
        if args.num_workers <= 1:
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
        else:
            chunk_size = math.ceil(args.games_per_iteration / args.num_workers)
            worker_args = []
            for w in range(args.num_workers):
                games_this_worker = min(
                    chunk_size,
                    args.games_per_iteration - w * chunk_size,
                )
                if games_this_worker <= 0:
                    continue
                worker_args.append(
                    (
                        w,
                        {k: v.cpu().clone() for k, v in bot.model.state_dict().items()},
                        games_this_worker,
                        args.board_size,
                        args.num_simulations,
                        args.agent_type,
                        args.mcts_batch_size,
                        args.c_puct,
                        args.self_play_temp,
                        args.temp_moves,
                        args.league_prob,
                        args.heuristic_prob,
                        league_path,
                        args.worker_device,
                        args.seed,
                        args.amp,
                        not args.no_compile,
                        not args.no_root_noise,
                        args.root_dirichlet_alpha,
                        args.root_dirichlet_epsilon,
                    )
                )
            with multiprocessing.Pool(args.num_workers) as pool:
                results = pool.starmap(_worker_self_play, worker_args)
            new_data = [item for data, _stats in results for item in data]
            self_play_stats = {
                "wins": sum(s["wins"] for _d, s in results),
                "losses": sum(s["losses"] for _d, s in results),
                "draws": sum(s["draws"] for _d, s in results),
                "games_league": sum(s["games_league"] for _d, s in results),
                "games_heuristic": sum(s["games_heuristic"] for _d, s in results),
                "games_self_play": sum(s["games_self_play"] for _d, s in results),
            }
            print(f"  Self-play  [{'=' * 25}] 100.0% ({args.games_per_iteration}/{args.games_per_iteration}) games")

        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
        buffer.extend(new_data)
        total_games = (
            self_play_stats["wins"] + self_play_stats["losses"] + self_play_stats["draws"]
        )
        game_success_rate = self_play_stats["wins"] / total_games if total_games else 0.0
        game_type_log = (
            f"league={self_play_stats['games_league']} "
            f"heuristic={self_play_stats['games_heuristic']} "
            f"self_play={self_play_stats['games_self_play']}"
        )
        if len(buffer) > buffer_max_size:
            buffer = buffer[-buffer_max_size:]

        # Train for several epochs on current buffer
        random.shuffle(buffer)
        total_pl = 0.0
        total_vl = 0.0
        n_batches = 0
        batches_per_epoch = max(1, len(buffer) // args.batch_size)
        for epoch in range(args.train_epochs):
            for start in range(0, len(buffer), args.batch_size):
                batch = buffer[start : start + args.batch_size]
                if len(batch) < 2:
                    continue
                log_train(epoch, args.train_epochs, start // args.batch_size, batches_per_epoch)
                pl, vl = train_step(
                    batch, bot.model, optimizer, args.board_size, device, args.value_coef
                )
                total_pl += pl
                total_vl += vl
                n_batches += 1
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        if n_batches > 0:
            avg_pl = total_pl / n_batches
            avg_vl = total_vl / n_batches
            avg_loss = avg_pl + args.value_coef * avg_vl
            print(
                f"  Loss: policy={avg_pl:.4f} value={avg_vl:.4f} buffer={len(buffer)} "
                f"game_success_rate={game_success_rate:.2f}  [{game_type_log}]"
            )
            if args.az_best_by == "loss" and avg_loss < best_loss:
                best_loss = avg_loss
                save_model_weights(bot.model, args.save_best_path)
                print(f"  -> saved best (by loss) to {args.save_best_path}")
        else:
            print(
                f"  Buffer size {len(buffer)} too small; skipping train  "
                f"game_success_rate={game_success_rate:.2f}  [{game_type_log}]"
            )

        if iteration % args.az_eval_freq == 0 or iteration == total_display:
            eval_random = evaluate_alphazero(
                bot=bot,
                board_size=args.board_size,
                num_games=args.eval_games,
                opponent="random",
            )
            n_heuristic = (
                args.az_eval_games_best
                if args.az_best_by == "heuristic"
                else args.eval_games
            )
            eval_heuristic = evaluate_alphazero(
                bot=bot,
                board_size=args.board_size,
                num_games=n_heuristic,
                opponent="heuristic",
            )
            print(
                f"  Eval: vs_random win_rate={eval_random['win_rate']:.2f}  "
                f"vs_heuristic win_rate={eval_heuristic['win_rate']:.2f}"
            )
            if (
                args.az_best_by == "heuristic"
                and eval_heuristic["win_rate"] > best_heuristic_wr
            ):
                best_heuristic_wr = eval_heuristic["win_rate"]
                save_model_weights(bot.model, args.save_best_path)
                print(f"  -> saved best (by heuristic) to {args.save_best_path}")

        # Save checkpoint every 5 iterations (and on last); used for league pool
        if iteration % 5 == 0 or iteration == total_display:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iteration}.pt")
            save_model_weights(bot.model, ckpt_path)
            checkpoint_pool.append(ckpt_path)
            checkpoint_pool = checkpoint_pool[-args.league_pool_size:]
            print(f"  -> checkpoint {ckpt_path}")

    print(progress_bar(total_display, total_display, width=30, prefix="Done.       ", suffix=""))
    print("AlphaZero training complete.")


def _run_hybrid_training(args: argparse.Namespace, device: torch.device) -> None:
    """Hybrid (CNN+Transformer) AlphaZero self-play and training loop."""
    if args.board_size != 9:
        raise ValueError("Hybrid architecture currently supports --board_size 9 only.")

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

    bot = HybridBot(
        model=model,
        board_size=args.board_size,
        device=device,
        num_simulations=args.num_simulations,
        compile_model=not args.no_compile,
        mcts_batch_size=args.mcts_batch_size,
        use_amp=args.amp,
    )
    optimizer = torch.optim.AdamW(
        bot.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.hybrid_weight_decay,
    )

    # Approximate total updates for cosine decay.
    est_positions_per_game = 45
    est_batches_per_epoch = max(1, math.ceil((args.games_per_iteration * est_positions_per_game) / args.batch_size))
    total_scheduler_steps = max(1, args.iterations * args.train_epochs * est_batches_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_scheduler_steps,
        eta_min=args.hybrid_lr_min,
    )

    buffer: list[tuple[np.ndarray, int, np.ndarray, float]] = []
    buffer_max_size = args.games_per_iteration * 200
    best_loss = float("inf")
    best_heuristic_wr = -1.0
    total_iterations = args.iterations
    checkpoint_pool: list[str] = []

    def log_self_play(current: int, total: int) -> None:
        msg = progress_bar(current, total, width=25, prefix="  Self-play ", suffix=" games")
        sys.stdout.write(f"\r{msg}")
        sys.stdout.flush()

    total_display = iteration_offset + total_iterations
    for i in range(total_iterations):
        iteration = iteration_offset + i + 1
        iter_msg = progress_bar(iteration, total_display, width=30, prefix="Iteration ", suffix="")
        print(iter_msg)

        league_bot: HybridBot | None = None
        if checkpoint_pool and args.league_prob > 0:
            league_path = random.choice(checkpoint_pool)
            league_model = AlphaZeroHybrid(board_size=args.board_size).to(device)
            load_weights(league_model, league_path, device)
            league_bot = HybridBot(
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
        total_pl = 0.0
        total_vl = 0.0
        n_batches = 0
        for _epoch in range(args.train_epochs):
            for start in range(0, len(buffer), args.batch_size):
                batch = buffer[start : start + args.batch_size]
                if len(batch) < 2:
                    continue
                pl, vl = train_step_hybrid(
                    batch=batch,
                    model=bot.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    board_size=args.board_size,
                    device=device,
                    value_coef=args.value_coef,
                    grad_clip_norm=args.hybrid_grad_clip_norm,
                )
                total_pl += pl
                total_vl += vl
                n_batches += 1

        total_games = (
            self_play_stats["wins"] + self_play_stats["losses"] + self_play_stats["draws"]
        )
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
            print(
                f"  Loss: policy={avg_pl:.4f} value={avg_vl:.4f} total={avg_loss:.4f} "
                f"buffer={len(buffer)} lr={optimizer.param_groups[0]['lr']:.6f} "
                f"game_success_rate={game_success_rate:.2f}  [{game_type_log}]"
            )
            if args.az_best_by == "loss" and avg_loss < best_loss:
                best_loss = avg_loss
                save_model_weights(bot.model, args.save_best_path)
                print(f"  -> saved best (by loss) to {args.save_best_path}")
        else:
            print(
                f"  Buffer size {len(buffer)} too small; skipping train  "
                f"game_success_rate={game_success_rate:.2f}  [{game_type_log}]"
            )

        if iteration % args.az_eval_freq == 0 or iteration == total_display:
            eval_random = evaluate_alphazero(
                bot=bot,
                board_size=args.board_size,
                num_games=args.eval_games,
                opponent="random",
            )
            n_heuristic = (
                args.az_eval_games_best
                if args.az_best_by == "heuristic"
                else args.eval_games
            )
            eval_heuristic = evaluate_alphazero(
                bot=bot,
                board_size=args.board_size,
                num_games=n_heuristic,
                opponent="heuristic",
            )
            print(
                f"  Eval: vs_random win_rate={eval_random['win_rate']:.2f}  "
                f"vs_heuristic win_rate={eval_heuristic['win_rate']:.2f}"
            )
            if (
                args.az_best_by == "heuristic"
                and eval_heuristic["win_rate"] > best_heuristic_wr
            ):
                best_heuristic_wr = eval_heuristic["win_rate"]
                save_model_weights(bot.model, args.save_best_path)
                print(f"  -> saved best (by heuristic) to {args.save_best_path}")

        if iteration % 5 == 0 or iteration == total_display:
            ckpt_path = os.path.join(args.checkpoint_dir, f"hybrid_checkpoint_{iteration}.pt")
            save_model_weights(bot.model, ckpt_path)
            checkpoint_pool.append(ckpt_path)
            checkpoint_pool = checkpoint_pool[-args.league_pool_size:]
            print(f"  -> checkpoint {ckpt_path}")

    print(progress_bar(total_display, total_display, width=30, prefix="Done.       ", suffix=""))
    print("Hybrid AlphaZero training complete.")


def _run_dqn_training(args: argparse.Namespace, device: torch.device) -> None:
    """DQN self-play and training loop."""
    model = DQN(board_size=args.board_size).to(device)
    target_model = DQN(board_size=args.board_size).to(device)
    target_model.load_state_dict(model.state_dict())

    dqn_save_best = os.path.join(args.checkpoint_dir, "dqn_best.pt")

    iteration_offset = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        load_weights(model, args.resume, device)
        target_model.load_state_dict(model.state_dict())
        print(f"Resumed from {args.resume}")
        base = os.path.basename(args.resume)
        if base.startswith("dqn_checkpoint_") and base.endswith(".pt"):
            try:
                iteration_offset = int(base[len("dqn_checkpoint_") : -len(".pt")])
            except ValueError:
                pass

    replay_buffer = ReplayBuffer(args.replay_buffer_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_heuristic_wr = -1.0
    total_iterations = args.iterations
    total_display = iteration_offset + total_iterations
    total_train_steps = 0
    checkpoint_pool: list[str] = []

    def log_self_play(current: int, total: int) -> None:
        msg = progress_bar(current, total, width=25, prefix="  Self-play ", suffix=" games")
        sys.stdout.write(f"\r{msg}")
        sys.stdout.flush()

    for i in range(total_iterations):
        iteration = iteration_offset + i + 1
        iter_msg = progress_bar(iteration, total_display, width=30, prefix="Iteration ", suffix="")
        print(iter_msg)

        epsilon = get_epsilon(
            global_step,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay_steps,
        )

        league_model = None
        if checkpoint_pool and args.league_prob > 0:
            league_path = random.choice(checkpoint_pool)
            if os.path.isfile(league_path):
                league_model = DQN(board_size=args.board_size).to(device)
                load_weights(league_model, league_path, device)
                league_model.eval()

        if args.heuristic_prob_start is not None and args.heuristic_prob_decay_iters is not None:
            t = min(1.0, (i + 1) / args.heuristic_prob_decay_iters)
            heuristic_prob_effective = args.heuristic_prob_start + t * (args.heuristic_prob - args.heuristic_prob_start)
        else:
            heuristic_prob_effective = args.heuristic_prob

        steps, wins, losses, draws, games_league, games_heuristic, games_self_play = dqn_self_play(
            model,
            replay_buffer,
            args.board_size,
            args.games_per_iteration,
            device,
            epsilon,
            heuristic_prob=heuristic_prob_effective,
            league_model=league_model,
            league_prob=args.league_prob,
            heuristic_win_bonus=args.heuristic_win_bonus,
            progress_callback=log_self_play,
            use_amp=args.amp,
        )
        global_step += steps
        total_games = wins + losses + draws
        game_success_rate = wins / total_games if total_games else 0.0
        game_type_log = f"league={games_league} heuristic={games_heuristic} self_play={games_self_play}"

        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        if len(replay_buffer) < args.batch_size:
            print(
                f"  Buffer size {len(replay_buffer)} < batch_size; skipping train  game_success_rate={game_success_rate:.2f}  [{game_type_log}]"
            )
        else:
            total_loss = 0.0
            n_train = 0
            for _ in range(args.train_steps_per_iteration):
                batch = replay_buffer.sample(
                    args.batch_size,
                    terminal_fraction=args.dqn_terminal_fraction,
                )
                loss = dqn_train_step(
                    batch,
                    model,
                    target_model,
                    optimizer,
                    args.board_size,
                    device,
                    gamma=args.gamma,
                    use_amp=args.amp,
                )
                total_loss += loss
                n_train += 1
                total_train_steps += 1
                if total_train_steps % args.target_update_freq == 0:
                    target_model.load_state_dict(model.state_dict())

            if n_train > 0:
                avg_loss = total_loss / n_train
                print(
                    f"  Loss: {avg_loss:.4f} buffer={len(replay_buffer)} epsilon={epsilon:.3f} game_success_rate={game_success_rate:.2f}  [{game_type_log}]"
                )
                if args.best_by == "loss" and avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_weights(model, dqn_save_best)
                    print(f"  -> saved best (by loss) to {dqn_save_best}")

        if iteration % args.eval_freq == 0 or iteration == total_display:
            eval_random = evaluate_dqn(
                model, args.board_size, device, args.eval_games, "random", use_amp=args.amp
            )
            n_heuristic = args.eval_games_best if args.best_by == "heuristic" else args.eval_games
            eval_heuristic = evaluate_dqn(
                model, args.board_size, device, n_heuristic, "heuristic", use_amp=args.amp
            )
            print(
                f"  Eval: vs_random win_rate={eval_random['win_rate']:.2f}  vs_heuristic win_rate={eval_heuristic['win_rate']:.2f}"
            )
            if args.best_by == "heuristic" and eval_heuristic["win_rate"] > best_heuristic_wr:
                best_heuristic_wr = eval_heuristic["win_rate"]
                save_model_weights(model, dqn_save_best)
                print(f"  -> saved best (by heuristic) to {dqn_save_best}")

        # Save checkpoint every 10 iterations (and on last)
        if iteration % 10 == 0 or iteration == total_display:
            ckpt_path = os.path.join(args.checkpoint_dir, f"dqn_checkpoint_{iteration}.pt")
            save_model_weights(model, ckpt_path)
            checkpoint_pool.append(ckpt_path)
            checkpoint_pool = checkpoint_pool[-args.league_pool_size:]
            print(f"  -> checkpoint {ckpt_path}")

    print(progress_bar(total_display, total_display, width=30, prefix="Done.       ", suffix=""))
    print("DQN training complete.")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
    
    main()
