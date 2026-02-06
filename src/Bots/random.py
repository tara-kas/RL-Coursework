from src.Bots.base_bot import BaseBot
from src.game_datatypes import GameState

import random

class Bot(BaseBot):
    def __init__(self):
        super().__init__()

    def move(self, game_state: GameState) -> tuple[int, int]:
        empty_cells = [(x, y) for x in range(game_state.board.shape[0]) 
                       for y in range(game_state.board.shape[1]) 
                       if game_state.board[x, y] == -1]
        
        if not empty_cells:
            return None

        chosen_cell = random.choice(empty_cells)
        
        return chosen_cell