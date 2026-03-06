import numpy as np


def preprocess_board(board: np.ndarray, current_player: int) -> np.ndarray:
    """
    Convert board to 3-plane representation. Returns shape (3, H, W) float.
    Plane 0: Current player's stones
    Plane 1: Opponent's stones
    Plane 2: Empty cells
    """
    plane_current = (board == current_player).astype(np.float32)
    plane_opponent = ((board != current_player) & (board != -1)).astype(np.float32)
    plane_empty = (board == -1).astype(np.float32)
    
    return np.stack([plane_current, plane_opponent, plane_empty], axis=0)