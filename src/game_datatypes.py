import numpy as np
        
class GameState():
    def __init__(self, grid_x: int, grid_y: int):
        self.board = np.zeros((grid_x, grid_y), dtype=int) - 1