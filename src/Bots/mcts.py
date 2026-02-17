from base_bot import BaseBot
from src.game_datatypes import GameState
from src.game_logic import GameLogic
import numpy as np

class MCTSNode:
    def __init__(self):
        self.visits = 0
        self.value_sum = 0
        self.children = None

    def get_value(self):
        return self.value_sum / self.visits

    def expand(self, child: MCTSNode, action):
        if action not in self.children:
            self.children[action] = child

    def get_child(self, child_index: int):
        return self.children[child_index]

    def get_ucb_score(self, actions):
        pass

    def simulate(self):
        """Play randomly until game state considered over"""
        pass

class PureMCTS(BaseBot):
    def __init__(self):
        super().__init__()
        self.root = MCTSNode(None)
        self.cur_node = self.root

    def move(self, **kwargs):
        t = kwargs.get("t", "random")
        if t == "random":
            return None
        else:
            return 

    def run(self):
        game = GameLogic(15,15,[{"type": "bot", "name": "mcts", "file": "mcts", "colour": (0,0,255)},{"type": "bot", "name": "mcts mirror", "file": "mcts", "colour": (0,255,0)}])
        
        #override the bots to be the same player
        game.bots["mcts"] = self
        game.bots["mcts mirror"] = self

        names = ["mcts", "mcts mirror"]

        self.cur_node = self.root

        #while we have two bots playing each other, both of them are the same
        while cur_node is not None: #while not at leaf
            next_action = game.get_bot_move(names[game.current_turn], t="ucb")
            game.check_valid_move(next_action)
            game.make_move(game.current_turn, next_action)
            game.next_turn()
            #play next action
            #check win?
            win = False
            loss = False
            if win == True:
                pass #backup value scores
            if loss == True:
                pass #backup value scores
                
            cur_node = cur_node.get_child(next_action)

        #we are now at a leaf
        #pick a random next action
        next_action = game.get_bot_move(names[game.current_turn], t="random")
        game.check_valid_move(next_action)
        game.make_move(game.current_turn, next_action)
        game.next_turn()
        cur_node.expand(next_action)

        game_finished = False
        while game_finished != True:
            #play randomly from both players
            next_action = game.get_bot_move(names[game.current_turn], t="random")
            game.check_valid_move(next_action)
            game.make_move(game.current_turn, next_action)
            game.next_turn()
        
        #backup value scores