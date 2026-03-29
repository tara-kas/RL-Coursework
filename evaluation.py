"""
Standalone evaluation: run N games with a model as player 0 vs an opponent (player 1).
Opponents: random, heuristic, DQN, or AlphaZero variants.
"""
import argparse
import random

import numpy as np
import torch

from src.gomoku_game import (
    GAME_NOT_OVER,
    WIN,
    apply_move,
    get_game_result,
    get_legal_moves,
)
from src.Bots.dqn import DQN, select_action as dqn_select_action
from src.Bots.alpha_zero_resnet import Bot as AlphaZeroResNetBot
from src.Bots.alpha_zero_transformer import Bot as AlphaZeroTransformerBot
from src.Bots.alpha_zero_hybrid import Bot as AlphaZeroHybridBot
from src.Bots.heuristic_tactical import predict as heuristic_predict
from src.model_loader import load_weights


def _build_agent_move_fn(
    agent_type: str,
    weights_path: str,
    board_size: int,
    device: torch.device,
    num_simulations: int,
    use_amp: bool,
):
    if agent_type == "dqn":
        model = DQN(board_size=board_size).to(device)
        load_weights(model, weights_path, device)
        model.eval()

        def get_move(board: np.ndarray, current_player: int):
            _, move = dqn_select_action(
                model,
                board,
                current_player,
                board_size,
                device,
                epsilon=0.0,
                use_amp=use_amp,
            )
            return move

        return get_move

    bot_cls_map = {
        "alphazero": AlphaZeroResNetBot,
        "alphazero-resnet": AlphaZeroResNetBot,
        "alphazero-transformer": AlphaZeroTransformerBot,
        "hybrid": AlphaZeroHybridBot,
        "alphazero-hybrid": AlphaZeroHybridBot,
    }
    if agent_type not in bot_cls_map:
        raise ValueError(f"Unsupported agent_type: {agent_type}")

    bot = bot_cls_map[agent_type](
        weights_path=weights_path,
        board_size=board_size,
        device=device,
        num_simulations=num_simulations,
        use_amp=use_amp,
    )

    def get_move(board: np.ndarray, current_player: int):
        return bot.predict(board, current_player=current_player)

    return get_move


def run_games(
    board_size: int,
    num_games: int,
    get_move_p0,
    get_move_p1,
) -> dict:
    """
    Run num_games with get_move_p0 as player 0 and get_move_p1 as player 1.
    get_move_*(board, current_player) -> (x, y).
    Returns dict with wins, losses, draws, win_rate, avg_return for player 0.
    """
    wins = 0
    losses = 0
    draws = 0
    for _ in range(num_games):
        board = np.full((board_size, board_size), -1, dtype=np.int32)
        current_player = 0
        while True:
            legal = get_legal_moves(board)
            if not legal:
                break
            if current_player == 0:
                move = get_move_p0(board, current_player)
            else:
                move = get_move_p1(board, current_player)
            if move not in legal:
                move = random.choice(legal)
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
    parser = argparse.ArgumentParser(
        description="Evaluate DQN or AlphaZero variants vs random, heuristic, DQN, or AlphaZero variants."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weights (DQN or AlphaZero depending on --agent_type)",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        choices=(
            "dqn",
            "alphazero",
            "alphazero-resnet",
            "alphazero-transformer",
            "hybrid",
            "alphazero-hybrid",
        ),
        default="dqn",
        help="Type of the model to evaluate (player 0)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=(
            "random",
            "heuristic",
            "dqn",
            "alphazero",
            "alphazero-resnet",
            "alphazero-transformer",
            "hybrid",
            "alphazero-hybrid",
        ),
        required=True,
        help="Opponent to play against (player 1)",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=1000,
        help="Number of games to run",
    )
    parser.add_argument(
        "--board_size",
        type=int,
        default=9,
        help="Board size (e.g. 9 or 15)",
    )
    parser.add_argument(
        "--opponent_weights",
        type=str,
        default=None,
        help="Path to opponent weights when --opponent is a learned agent (required then)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu); default auto",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use FP16 autocast for inference",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=50,
        help="MCTS simulations per move when model or opponent is an AlphaZero variant",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    learned_opponents = {
        "dqn",
        "alphazero",
        "alphazero-resnet",
        "alphazero-transformer",
        "hybrid",
        "alphazero-hybrid",
    }
    if args.opponent in learned_opponents and not args.opponent_weights:
        parser.error(f"--opponent {args.opponent} requires --opponent_weights")

    # Player 0: model under evaluation
    get_move_p0 = _build_agent_move_fn(
        agent_type=args.agent_type,
        weights_path=args.model,
        board_size=args.board_size,
        device=device,
        num_simulations=args.num_simulations,
        use_amp=args.amp,
    )

    # Player 1: opponent
    if args.opponent == "random":
        def get_move_p1(board: np.ndarray, current_player: int):
            legal = get_legal_moves(board)
            return random.choice(legal)
    elif args.opponent == "heuristic":
        def get_move_p1(board: np.ndarray, current_player: int):
            return heuristic_predict(board.copy(), current_player)
    else:
        get_move_p1 = _build_agent_move_fn(
            agent_type=args.opponent,
            weights_path=args.opponent_weights,
            board_size=args.board_size,
            device=device,
            num_simulations=args.num_simulations,
            use_amp=args.amp,
        )

    print(
        f"Evaluating {args.agent_type} ({args.model}) vs {args.opponent}"
        + (f" ({args.opponent_weights})" if args.opponent in learned_opponents else "")
        + f" over {args.num_games} games (board_size={args.board_size})..."
    )
    result = run_games(
        args.board_size,
        args.num_games,
        get_move_p0,
        get_move_p1,
    )
    print(
        f"  Wins: {result['wins']}  Losses: {result['losses']}  Draws: {result['draws']}"
    )
    print(f"  Win rate (player 0): {result['win_rate']:.4f}")
    print(f"  Avg return (player 0): {result['avg_return']:.4f}")


if __name__ == "__main__":
    main()
