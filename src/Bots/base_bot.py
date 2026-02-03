from src.game_datatypes import GameState, Action

class BaseBot():
    def __init__(self, name:str):
        self.name = name

    def move(self, game_state:GameState) -> Action:
        pass