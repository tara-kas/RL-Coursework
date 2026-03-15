"""
Standalone evaluation: run N games with a model as player 0 vs an opponent (player 1).
Opponents: random, heuristic, alphazero.
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
from src.Bots.alpha_zero_transform import Bot as AlphaZeroBot
from src.Bots.heuristic_tactical import predict as heuristic_predict
from src.model_loader import load_weights


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
        description="Evaluate a DQN or AlphaZero model vs random, heuristic, or AlphaZero."
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
        choices=("dqn", "alphazero"),
        default="dqn",
        help="Type of the model to evaluate (player 0)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=("random", "heuristic", "alphazero"),
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
        help="Path to AlphaZero weights when --opponent alphazero (required then)",
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
        help="MCTS simulations per move when model or opponent is AlphaZero",
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

    if args.opponent == "alphazero" and not args.opponent_weights:
        parser.error("--opponent alphazero requires --opponent_weights")

    # Player 0: model under evaluation
    if args.agent_type == "dqn":
        model_p0 = DQN(board_size=args.board_size).to(device)
        load_weights(model_p0, args.model, device)
        model_p0.eval()

        def get_move_p0(board: np.ndarray, current_player: int):
            _, move = dqn_select_action(
                model_p0,
                board,
                current_player,
                args.board_size,
                device,
                epsilon=0.0,
                use_amp=args.amp,
            )
            return move
    else:
        bot_p0 = AlphaZeroBot(
            weights_path=args.model,
            board_size=args.board_size,
            device=device,
            num_simulations=args.num_simulations,
            use_amp=args.amp,
        )

        def get_move_p0(board: np.ndarray, current_player: int):
            return bot_p0.predict(board, current_player=current_player)

    # Player 1: opponent
    if args.opponent == "random":
        def get_move_p1(board: np.ndarray, current_player: int):
            legal = get_legal_moves(board)
            return random.choice(legal)
    elif args.opponent == "heuristic":
        def get_move_p1(board: np.ndarray, current_player: int):
            return heuristic_predict(board.copy(), current_player)
    else:
        bot_p1 = AlphaZeroBot(
            weights_path=args.opponent_weights,
            board_size=args.board_size,
            device=device,
            num_simulations=args.num_simulations,
            use_amp=args.amp,
        )

        def get_move_p1(board: np.ndarray, current_player: int):
            return bot_p1.predict(board, current_player=current_player)

    print(
        f"Evaluating {args.agent_type} ({args.model}) vs {args.opponent}"
        + (f" ({args.opponent_weights})" if args.opponent == "alphazero" else "")
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
