import numpy as np
import torch

def preprocess_board(board: np.ndarray, current_player: int) -> torch.Tensor:
        # Current player's stones
        plane_current = (board == current_player).astype(float)

        # Opponent's stones
        plane_opponent = ((board != current_player) & (board != -1)).astype(float)

        # Empty squares
        plane_empty = (board == -1).astype(float)
        
        return torch.tensor([plane_current, plane_opponent, plane_empty]).float()