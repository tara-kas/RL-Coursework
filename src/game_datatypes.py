import numpy as np

class Action():
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        
class GameState():
    def __init__(self, grid_x: int, grid_y: int):
        self.board = np.zeros((grid_x, grid_y), dtype=int)