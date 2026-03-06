import numpy as np


# Game result constants for get_game_result()
GAME_NOT_OVER = -1
DRAW = 0
WIN = 1


def get_game_result(
    board: np.ndarray, last_move: tuple[int, int], last_player: int
) -> int:
    """
    Stateless outcome after last_move by last_player.
    Returns: WIN (1) if last_player has five-in-a-row, DRAW (0) if board full and no win,
    GAME_NOT_OVER (-1) otherwise.
    """
    h, w = board.shape
    new_x, new_y = last_move

    # Five-in-a-row check (same 4 directions as game_logic.five_in_a_row)
    count = 0
    for cell in range(9):
        if 0 <= new_x - 4 + cell < h:
            if board[new_x - 4 + cell, new_y] == last_player:
                count += 1
                if count == 5:
                    return WIN
            else:
                count = 0

    count = 0
    for cell in range(9):
        if 0 <= new_y - 4 + cell < w:
            if board[new_x, new_y - 4 + cell] == last_player:
                count += 1
                if count == 5:
                    return WIN
            else:
                count = 0

    count = 0
    for cell in range(9):
        if 0 <= new_x - 4 + cell < h and 0 <= new_y - 4 + cell < w:
            if board[new_x - 4 + cell, new_y - 4 + cell] == last_player:
                count += 1
                if count == 5:
                    return WIN
            else:
                count = 0

    count = 0
    for cell in range(9):
        if 0 <= new_x + 4 - cell < h and 0 <= new_y - 4 + cell < w:
            if board[new_x + 4 - cell, new_y - 4 + cell] == last_player:
                count += 1
                if count == 5:
                    return WIN
            else:
                count = 0

    if not np.any(board == -1):
        return DRAW
    return GAME_NOT_OVER


def get_legal_moves(board: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (x, y) positions that are empty."""
    return [(x, y)
        for x in range(board.shape[0])
        for y in range(board.shape[1])
        if board[x, y] == -1
    ]


def apply_move(board: np.ndarray, move: tuple[int, int], player: int) -> np.ndarray:
    """Return a new board with the move applied. Original board is not modified."""
    
    new_board = board.copy()
    x, y = move
    new_board[x, y] = player

    return new_board


def is_board_full(board: np.ndarray) -> bool:
    """True only when board is full (no empty cells)."""

    return not np.any(board == -1)
