"""
Heuristic tactical bot for Gomoku: wins when possible, blocks opponent threats,
builds toward 5-in-a-row. Used as an opponent during training.
"""
import random

import numpy as np

from src.Bots.base_bot import BaseBot
from src.game_datatypes import GameState
from src.gomoku_game import WIN, get_legal_moves, apply_move, get_game_result


# Four directions for line detection (horizontal, vertical, both diagonals)
_DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]


def _count_ray(
    board: np.ndarray,
    start_x: int,
    start_y: int,
    dx: int,
    dy: int,
    player: int,
) -> int:
    """Count consecutive stones of player from (start_x, start_y) in direction (dx, dy)."""
    h, w = board.shape
    n = 0
    x, y = start_x, start_y
    while 0 <= x < h and 0 <= y < w and board[x, y] == player:
        n += 1
        x += dx
        y += dy
    return n


def _line_length_if_play(
    board: np.ndarray,
    x: int,
    y: int,
    dx: int,
    dy: int,
    player: int,
) -> int:
    """If player plays at (x,y), return the length of the line through (x,y) in direction (dx,dy)."""
    assert board[x, y] == -1
    forward = _count_ray(board, x + dx, y + dy, dx, dy, player)
    back = _count_ray(board, x - dx, y - dy, -dx, -dy, player)
    return 1 + forward + back


def _max_line_if_play(board: np.ndarray, x: int, y: int, player: int) -> int:
    """Maximum line length (over 4 directions) if player plays at (x, y)."""
    if board[x, y] != -1:
        return 0
    best = 0
    for dx, dy in _DIRS:
        length = _line_length_if_play(board, x, y, dx, dy, player)
        if length > best:
            best = length
    return best


def predict(board: np.ndarray, current_player: int) -> tuple[int, int]:
    """
    Return the next move (x, y) for current_player.
    Priority: 1) win (5), 2) block opponent 4, 3) make our 4, 4) block opponent 3,
    5) make our 3, 6) best building move (max line length), 7) random.
    """
    legal = get_legal_moves(board)
    if not legal:
        raise ValueError("No legal moves")

    opponent = 1 - current_player

    # complete 5
    for move in legal:
        new_board = apply_move(board, move, current_player)
        if get_game_result(new_board, move, current_player) == WIN:
            return move

    # block opponent's 4-in-a-row
    for move in legal:
        new_board = apply_move(board, move, opponent)
        if get_game_result(new_board, move, opponent) == WIN:
            return move

    # make 4 in a row
    best_4 = [m for m in legal if _max_line_if_play(board, m[0], m[1], current_player) >= 4]
    if best_4:
        return random.choice(best_4)

    # block opponent's 3 in a row
    best_block_3 = [
        m for m in legal
        if _max_line_if_play(board, m[0], m[1], opponent) >= 3
    ]
    if best_block_3:
        return random.choice(best_block_3)

    # make 3 in a row
    scores_own = [(_max_line_if_play(board, m[0], m[1], current_player), m) for m in legal]
    max_own = max(s[0] for s in scores_own)
    if max_own >= 3:
        best_3 = [m for score, m in scores_own if score >= 3]
        return random.choice(best_3)

    # best move to build line length
    if scores_own:
        scores_own.sort(key=lambda s: (-s[0], s[1]))
        return scores_own[0][1]

    # otherwise random
    return random.choice(legal)


class Bot(BaseBot):
    def __init__(self):
        super().__init__()

    def move(self, game_state: GameState) -> tuple[int, int] | None:
        legal = get_legal_moves(game_state.board)
        if not legal:
            return None

        current_player = game_state.current_player
        if current_player is None:
            count_0 = int(np.sum(game_state.board == 0))
            count_1 = int(np.sum(game_state.board == 1))
            current_player = 0 if count_0 <= count_1 else 1

        return predict(game_state.board.copy(), int(current_player))
