import random

class GameLogic():
    def __init__(self):
        self.users = [{"type": "player", "name": "Test Player"}, {"type": "bot", "name": "Mr Random", "file": "random"}]

        self.current_turn = random.randint(0, len(self.users) - 1)
        
    def make_move(self, user_index:int, position:tuple[int, int]):
        pass
        
    def next_turn(self):
        self.current_turn = (self.current_turn + 1) % len(self.users)
        