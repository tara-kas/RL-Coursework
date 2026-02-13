from abc import ABC, abstractmethod

from src.game_datatypes import GameState

class BaseBot(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def move(self, game_state:GameState) -> tuple[int, int] | None:
        pass