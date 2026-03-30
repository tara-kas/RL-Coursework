import random
import importlib
import inspect
import numpy as np

from src.game_datatypes import GameState

class GameLogic():
    def __init__(self, grid_x:int=15, grid_y:int=15, users:list[dict]=None):
        self.grid_x = grid_x
        self.grid_y = grid_y
        if users is None:
            self.users = [{"type": "player", "name": "Test Player", "colour": (0,0,255)}, {"type": "bot", "name": "Mr Random", "file": "random", "colour": (255,0,0)}]
        else:
            self.users = users

        self.bots = {}

        for user in self.users:
            if user["type"] == "bot":
                self.bots[user["name"]] = self.load_bot(user["file"], user)
        
        self.game_state = GameState(grid_x, grid_y)
        self.move_mask = np.zeros((grid_x, grid_y), dtype=object)

        for i in range(grid_x):
            for j in range(grid_y):
                self.move_mask[i][j] = (i,j)

        self.current_turn = random.randint(0, len(self.users) - 1)
        if hasattr(self.game_state, "current_player"):
            self.game_state.current_player = self.current_turn
        self.game_over = False
        self.winner = None

    def reset_game(self):
        self.game_state = GameState(self.grid_x, self.grid_y)
        self.current_turn = random.randint(0, len(self.users) - 1)
        if hasattr(self.game_state, "current_player"):
            self.game_state.current_player = self.current_turn
        self.game_over = False
        self.winner = None

    def _board(self) -> np.ndarray:
        # Backward compatibility: older tests may assign a raw ndarray to game_state.
        return self.game_state.board if hasattr(self.game_state, "board") else self.game_state

    def check_valid_move(self, position:tuple[int, int] | None) -> bool:
        if position is None:
            return False
        if not isinstance(position, tuple) or len(position) != 2:
            return False

        x, y = position
        board = self._board()

        if x < 0 or y < 0 or x >= self.grid_x or y >= self.grid_y:
            return False

        return board[x, y] == -1
    
    def get_valid_moves(self) -> list:
        board = self._board()
        res = np.where(board == -1, self.move_mask, 0)
        valid = np.nonzero(res)
        posified = [(x,y) for x,y in zip(valid[0], valid[1])]
        return posified
        
    def make_move(self, user_index:int, position:tuple[int, int]) -> None:
        if self.game_over:
            return

        x, y = position
        if not self.check_valid_move(position):
            return

        self._board()[x, y] = user_index
            
        if self.five_in_a_row(x, y, user_index):
            print(f"player {user_index} has won!!!!!!!!!")
            self.game_over = True
            self.winner = user_index
            return

        if not np.any(self._board() == -1):
            self.game_over = True
            self.winner = None
            

    def next_turn(self):
        self.current_turn = (self.current_turn + 1) % len(self.users)
        if hasattr(self.game_state, "current_player"):
            self.game_state.current_player = self.current_turn

    def load_bot(self, file_name: str, user: dict | None = None) -> object:
        # Load the bot module from the file name
        try:
            module = importlib.import_module(f"src.Bots.{file_name}")
            
            # Create instance of bot
            if hasattr(module, "Bot"):
                bot_class = getattr(module, "Bot")
                kwargs = user.get("bot_kwargs", {}) if user else {}
                if kwargs:
                    try:
                        sig = inspect.signature(bot_class.__init__)
                        params = sig.parameters
                        accepts_var_kwargs = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in params.values()
                        )
                        if not accepts_var_kwargs:
                            kwargs = {
                                k: v for k, v in kwargs.items() if k in params
                            }
                    except (TypeError, ValueError):
                        pass

                instance = bot_class(**kwargs)
                
                return instance
            else:
                print(f"No Bot class found in {file_name}.py")
                return None
            
        except ImportError as exc:
            print(f"Failed to import bot module '{file_name}': {exc}")
                
        return None
    
    def get_bot_move(self, bot_name:str, **kwargs) -> tuple[int, int]:
        if bot_name not in self.bots:
            raise KeyError(f"Bot {bot_name} not found.")

        bot = self.bots[bot_name]
        if bot is None:
            return None
        if hasattr(self.game_state, "current_player"):
            self.game_state.current_player = self.current_turn

        if hasattr(bot, "move"):
            move = bot.move(self.game_state, **kwargs)
            return move
        return None
        
    def five_in_a_row(self, new_x, new_y, player):
        board = self._board()
        count = 0
        
        # left to right
        for cell in range(9):
            if new_x-4+cell >= 0 and new_x-4+cell < self.grid_x:
                if board[new_x - 4 + cell][new_y] == player:
                    count = count + 1
                    if count == 5:
                        return True
                else:
                    count = 0
        
        count = 0        
        
        # up down
        for cell in range(9):
            if new_y - 4 + cell >= 0 and new_y - 4 + cell < self.grid_y:
                if board[new_x][new_y - 4 + cell] == player:
                    count+=1
                    if count == 5:
                        return True
                else:
                    count = 0
                
        count = 0   
                
        # diagonal top left to bottom right
        for cell in range (9):
            if new_x - 4 + cell >= 0 and new_x - 4 + cell < self.grid_x and new_y - 4 + cell >= 0 and new_y - 4 + cell < self.grid_y:
                if board[new_x - 4 + cell][new_y - 4 + cell] == player:
                    count+=1
                    if count == 5:
                        return True
                else:
                    count = 0
                
        count = 0   
        
        # diagonal top right to bottom left
        for cell in range (9):
            if new_x + 4 - cell >= 0 and new_x + 4 - cell < self.grid_x and new_y - 4 + cell >= 0 and new_y - 4 + cell < self.grid_y:
                if board[new_x + 4 - cell][new_y - 4 + cell] == player:
                    count+=1
                    if count == 5:
                        return True
                else:
                    count = 0

        count = 0 
            
        return False
