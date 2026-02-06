import random

from src.game_datatypes import GameState

class GameLogic():
    def __init__(self, grid_x:int=15, grid_y:int=15, users:list[dict]=None):
        if users is None:
            self.users = [{"type": "player", "name": "Test Player", "colour": (0,0,255)}, {"type": "bot", "name": "Mr Random", "file": "random", "colour": (255,0,0)}]
        else:
            self.users = users
        
        self.game_state = GameState(grid_x, grid_y)

        self.current_turn = random.randint(0, len(self.users) - 1)

    def check_valid_move(self, position:tuple[int, int]):
        x, y = position

        return self.game_state.board[x, y] == -1
        
    def make_move(self, user_index:int, position:tuple[int, int]):
        if self.users[user_index]["type"] == "player":
            if self.check_valid_move(position):
                x, y = position
                self.game_state.board[x, y] = user_index
        
    def next_turn(self):
        self.current_turn = (self.current_turn + 1) % len(self.users)
        