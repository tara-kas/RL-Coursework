"""
Heuristic tactical bot for Gomoku: wins when possible, blocks opponent's 4-in-a-row, else random.
Used as an opponent during training to inject defensive/winning situations.
"""
import random

import numpy as np

from src.gomoku_game import WIN, get_legal_moves, apply_move, get_game_result


def predict(board: np.ndarray, current_player: int) -> tuple[int, int]:
    """
    Return the next move (x, y) for current_player.
    Priority: 1) winning move (complete 5), 2) block opponent's 4, 3) random legal move.
    """
    legal = get_legal_moves(board)
    if not legal:
        raise ValueError("No legal moves")

    # Priority 1: winning move
    for move in legal:
        new_board = apply_move(board, move, current_player)
        if get_game_result(new_board, move, current_player) == WIN:
            return move

    # Priority 2: block opponent's 4-in-a-row (opponent would win by playing there)
    opponent = 1 - current_player
    for move in legal:
        new_board = apply_move(board, move, opponent)
        if get_game_result(new_board, move, opponent) == WIN:
            return move

    # Priority 3: random legal move
    return random.choice(legal)
