import numpy as np
import torch

from src.gomoku_game import apply_move, get_legal_moves, is_board_full
from src.gomoku_utils import preprocess_board
import torch.nn as nn


class MCTSNode:
    def __init__(self, board: np.ndarray, current_player: int, parent: "MCTSNode | None" = None):
        self.board = board
        self.current_player = current_player
        self.parent = parent
        self.children: dict[tuple[int, int], MCTSNode] = {}
        self.N = 0
        self.W = 0.0  # Total value from this node's player perspective
        self.P: dict[tuple[int, int], float] = {}

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0


def run_mcts(board: np.ndarray, current_player: int, model: nn.Module, board_size: int, num_simulations: int) -> tuple[int, int]:
    pass