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
)
from src.gomoku_utils import preprocess_board
from src.Bots.alpha_zero_transform import AlphaZeroTransform, Bot
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
    bot: Bot,
    board_size: int,
    num_games: int,
    c_puct: float = 1.5,
    self_play_temp: float = 1.0,
    temp_moves: int = 30,
    progress_callback: Callable[[int, int], None] | None = None,
    league_bot: Bot | None = None,
    league_prob: float = 0.0,
    heuristic_prob: float = 0.0,
) -> list[tuple[np.ndarray, int, np.ndarray, float]]:
    """
    Run num_games. Each game is self-play, league (vs league_bot), or heuristic (vs tactical bot)
    with probabilities (1 - league_prob - heuristic_prob), league_prob, heuristic_prob.
    Returns list of (board, current_player, policy, z).
    """
    bot.model.eval()
    buffer: list[tuple[np.ndarray, int, np.ndarray, float]] = []

    for g in range(num_games):
        if progress_callback is not None:
            progress_callback(g + 1, num_games)
        r = random.random()
        if league_bot is not None and r < league_prob:
            opponent = league_bot
        elif r < league_prob + heuristic_prob:
            opponent = heuristic_predict
        else:
            opponent = None

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

    return buffer


def _worker_self_play(
    worker_id: int,
    state_dict: dict,
    num_games: int,
    board_size: int,
    num_simulations: int,
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
) -> list[tuple[np.ndarray, int, np.ndarray, float]]:
    """
    Worker for parallel self-play. Loads model from state_dict, runs num_games, returns buffer.
    Must be top-level for pickling in multiprocessing.
    """
    if seed_base is not None:
        random.seed(seed_base + worker_id)
        np.random.seed(seed_base + worker_id)
        torch.manual_seed(seed_base + worker_id)

    device = torch.device(device_str)
    model = AlphaZeroTransform(board_size=board_size).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    bot = Bot(
        model=model,
        board_size=board_size,
        device=device,
        num_simulations=num_simulations,
        compile_model=compile_model,
        mcts_batch_size=mcts_batch_size,
        use_amp=use_amp,
    )

    league_bot: Bot | None = None
    if league_path and league_prob > 0 and os.path.isfile(league_path):
        league_model = AlphaZeroTransform(board_size=board_size).to(device)
        load_weights(league_model, league_path, device)
        league_bot = Bot(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Gomoku self-play training (AlphaZero or DQN)")
    parser.add_argument(
        "--agent_type",
        type=str,
        default="alphazero",
        choices=["alphazero", "dqn"],
        help="Agent type: alphazero (MCTS + policy/value) or dqn (Q-network + replay buffer)",
    )
    parser.add_argument("--board_size", type=int, default=15)
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--games_per_iteration", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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
    # DQN-specific (ignored when agent_type == alphazero)
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

    if args.agent_type == "alphazero":
        _run_alphazero_training(args, device)
    elif args.agent_type == "dqn":
        _run_dqn_training(args, device)
    else:
        raise ValueError(f"Unknown agent_type: {args.agent_type}")


def _run_alphazero_training(args: argparse.Namespace, device: torch.device) -> None:
    """AlphaZero self-play and training loop."""
    model = AlphaZeroTransform(board_size=args.board_size).to(device)
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

    bot = Bot(
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
        league_bot: Bot | None = None
        if checkpoint_pool and args.league_prob > 0:
            league_path = random.choice(checkpoint_pool)
            if args.num_workers == 1:
                league_model = AlphaZeroTransform(board_size=args.board_size).to(device)
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

        # Self-play: parallel workers or single process
        if args.num_workers <= 1:
            new_data = self_play(
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
                    )
                )
            with multiprocessing.Pool(args.num_workers) as pool:
                results = pool.starmap(_worker_self_play, worker_args)
            new_data = [item for sublist in results for item in sublist]
            print(f"  Self-play  [{'=' * 25}] 100.0% ({args.games_per_iteration}/{args.games_per_iteration}) games")

        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
        buffer.extend(new_data)
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
                f"  Loss: policy={avg_pl:.4f} value={avg_vl:.4f} buffer={len(buffer)}"
            )
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model_weights(bot.model, args.save_best_path)
                print(f"  -> saved best to {args.save_best_path}")

        # Save checkpoint every 5 iterations (and on last); used for league pool
        if iteration % 5 == 0 or iteration == total_display:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iteration}.pt")
            save_model_weights(bot.model, ckpt_path)
            checkpoint_pool.append(ckpt_path)
            checkpoint_pool = checkpoint_pool[-args.league_pool_size:]
            print(f"  -> checkpoint {ckpt_path}")

    print(progress_bar(total_display, total_display, width=30, prefix="Done.       ", suffix=""))
    print("AlphaZero training complete.")


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
    total_iterations = args.iterations
    total_display = iteration_offset + total_iterations
    total_train_steps = 0

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

        steps, wins, losses, draws = dqn_self_play(
            model,
            replay_buffer,
            args.board_size,
            args.games_per_iteration,
            device,
            epsilon,
            heuristic_prob=args.heuristic_prob,
            progress_callback=log_self_play,
            use_amp=args.amp,
        )
        global_step += steps
        total_games = wins + losses + draws
        game_success_rate = wins / total_games if total_games else 0.0

        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        if len(replay_buffer) < args.batch_size:
            print(
                f"  Buffer size {len(replay_buffer)} < batch_size; skipping train  game_success_rate={game_success_rate:.2f}"
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
                    f"  Loss: {avg_loss:.4f} buffer={len(replay_buffer)} epsilon={epsilon:.3f} game_success_rate={game_success_rate:.2f}"
                )
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_weights(model, dqn_save_best)
                    print(f"  -> saved best to {dqn_save_best}")

        if iteration % args.eval_freq == 0 or iteration == total_display:
            eval_random = evaluate_dqn(
                model, args.board_size, device, args.eval_games, "random", use_amp=args.amp
            )
            eval_heuristic = evaluate_dqn(
                model, args.board_size, device, args.eval_games, "heuristic", use_amp=args.amp
            )
            print(
                f"  Eval: vs_random win_rate={eval_random['win_rate']:.2f}  vs_heuristic win_rate={eval_heuristic['win_rate']:.2f}"
            )

        # Save checkpoint every 10 iterations (and on last)
        if iteration % 10 == 0 or iteration == total_display:
            ckpt_path = os.path.join(args.checkpoint_dir, f"dqn_checkpoint_{iteration}.pt")
            save_model_weights(model, ckpt_path)
            print(f"  -> checkpoint {ckpt_path}")

    print(progress_bar(total_display, total_display, width=30, prefix="Done.       ", suffix=""))
    print("DQN training complete.")


if __name__ == "__main__":
    main()
