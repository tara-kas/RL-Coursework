from src.game_datatypes import GameState

class BaseBot():
    def __init__(self):
        pass

    def move(self, game_state:GameState) -> tuple[int, int]:
        pass