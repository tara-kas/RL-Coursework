from src.Bots.base_bot import BaseBot
from src.game_datatypes import GameState, Action

import random

class Bot(BaseBot):
    def __init__(self, name:str="Mr Random"):
        super().__init__(name)

    def move(self, game_state: GameState) -> Action:
        empty_cells = [(x, y) for x in range(game_state.board.shape[0]) 
                       for y in range(game_state.board.shape[1]) 
                       if game_state.board[x, y] == 0]
        
        if not empty_cells:
            return None

        chosen_cell = random.choice(empty_cells)
        
        return Action(chosen_cell[0], chosen_cell[1])