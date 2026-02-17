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

    def run(self):
        cur_node = self.root
        while cur_node is not None: #while not at leaf
            next_priors = cur_node.get_ucb_scores()
            next_action = np.argmax(next_priors)
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
        next_action = np.random.randint(1,225)
        cur_node.expand(next_action)

        game_finished = False
        while game_finished != True:
            next_action = np.random.randint(1,255)
            #play randomly from both players
        
        #backup value scores