import random
import importlib
import numpy as np

from src.game_datatypes import GameState
from .scene_manager import SceneManager

class GameLogic():
    def __init__(self, grid_x:int=15, grid_y:int=15, users:list[dict]=None):
        if users is None:
            self.users = [{"type": "player", "name": "Test Player", "colour": (0,0,255)}, {"type": "bot", "name": "Mr Random", "file": "random", "colour": (255,0,0)}]
        else:
            self.users = users

        self.bots = {}

        for user in self.users:
            if user["type"] == "bot":
                self.bots[user["name"]] = self.load_bot(user["file"])
        
        self.game_state = GameState(grid_x, grid_y)

        self.current_turn = random.randint(0, len(self.users) - 1)

    def check_valid_move(self, position:tuple[int, int]) -> bool:
        x, y = position

        return self.game_state.board[x, y] == -1
        
    def make_move(self, user_index:int, position:tuple[int, int]) -> None:
        x, y = position
        if self.check_valid_move(position):
            self.game_state.board[x, y] = user_index
            
        if self.five_in_a_row(x,y,user_index):
            print(f"player {user_index} has won!!!!!!!!!")
            SceneManager.shutdown()
            

    def next_turn(self):
        self.current_turn = (self.current_turn + 1) % len(self.users)

    def load_bot(self, file_name:str) -> object:
        # Load the bot module from the file name
        try:
            module = importlib.import_module(f"src.Bots.{file_name}")
            
            # Create instance of bot
            if hasattr(module, "Bot"):
                bot_class = getattr(module, "Bot")
                instance = bot_class()
                
                return instance
            else:
                print(f"No Bot class found in {file_name}.py")
                return None
            
        except ImportError:
            print(f"No module named {file_name} found in Bots directory.")
                
        return None
    
    def get_bot_move(self, bot_name:str) -> tuple[int, int]:
        if bot_name not in self.bots:
            raise KeyError(f"Bot {bot_name} not found.")

        bot = self.bots[bot_name]

        if hasattr(bot, "move"):
            move = bot.move(self.game_state)
            return move
        
    def five_in_a_row(self, new_x, new_y, player):
        count = 0
        
        # left to right
        for cell in range(9):
            try: 
                if self.game_state.board[new_x - 4 + cell][new_y] == player:
                    count+=1
            except IndexError:
                pass
            else:
                count = 0
        
        if count == 5:
            return True
        else:
            count = 0        
        
        # up down
        for cell in range(9):
            try:
                if self.game_state.board[new_x][new_y - 4 + cell] == player:
                    count+=1
            except IndexError:
                pass
            else:
                count = 0
                
        if count == 5:
            return True
        else:
            count = 0   
                
        # diagonal top left to bottom right
        for cell in range (9):
            try:
                if self.game_state.board[new_x - 4 + cell][new_y - 4 + cell]:
                    count+=1
            except IndexError:
                pass
            else:
                count = 0
                
        if count == 5:
            return True
        else:
            count = 0   
        
        # diagonal top right to bottom left
        for cell in range (9):
            try:
                if self.game_state.board[new_x + 4 - cell][new_y - 4 + cell]:
                    count+=1
            except IndexError:
                pass
            else:
                count = 0
                
        if count == 5:
            return True
        else:
            count = 0   
            
        print(f"{new_x}, {new_y}")
            
        return False