import numpy as np


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
