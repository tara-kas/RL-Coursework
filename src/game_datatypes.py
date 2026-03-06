import numpy as np


class GameState():
    def __init__(self, grid_x: int, grid_y: int, current_player: int | None = None):
        self.board = np.zeros((grid_x, grid_y), dtype=int) - 1
        self.current_player = current_player