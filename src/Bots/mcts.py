from src.Bots.base_bot import BaseBot
from src.game_datatypes import GameState
from src.game_logic import GameLogic
import numpy as np

#TODO:
#upper confidence bound method

class MCTSNode:
    def __init__(self, state: GameState):
        self.visits = 0
        self.value_sum = 0
        self.children = None
        self.state = state
        self.last_player = -1

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

class Bot(BaseBot):
    def __init__(self):
        super().__init__()
        self.game = GameLogic(15,15,[{"type": "bot", "name": "mcts", "file": "mcts", "colour": (0,0,255)},{"type": "bot", "name": "mcts mirror", "file": "mcts", "colour": (0,255,0)}])
        self.root = MCTSNode(self.game.game_state)
        self.cur_node = self.root

    def move(self, game_state, **kwargs):
        t = kwargs.get("t", "random")
        moves = self.game.get_valid_moves()
        if len(moves) == 0:
            return None
        if t == "random":
            choice = np.random.randint(0,len(moves))
            return moves[choice]
        else:
            children = [child for child in self.cur_node.children.values()] #get states into a list
            scores = [child.get_value() for child in children] #get value for each child
            ucb = np.argmax(scores) #pick highest value greedily
            desired_state = children[ucb].state
            current_state = self.cur_node.state
            print(desired_state)
            print(current_state)
            #extract move from difference between states
            return moves[0]

        
    def backup(self, search_path, winner):
        for node in search_path:
            node.visits += 1
            if node.last_player == winner:
                node.value_sum += 1

    def run(self):
        self.game = GameLogic(15,15,[{"type": "bot", "name": "mcts", "file": "mcts", "colour": (0,0,255)},{"type": "bot", "name": "mcts mirror", "file": "mcts", "colour": (0,255,0)}])
        
        #override the bots to be the same player
        self.game.bots["mcts"] = self
        self.game.bots["mcts mirror"] = self

        names = ["mcts", "mcts mirror"]
        winner = -1

        self.cur_node = self.root
        search_path = []

        #while we have two bots playing each other, both of them are the same
        while self.cur_node is not None: #while not at leaf
            search_path.append(self.cur_node)
            next_action = self.game.get_bot_move(names[self.game.current_turn], t="ucb")
            if next_action is None:
                self.backup(search_path, -1) #draw reached on last move
                return
            self.game.check_valid_move(next_action)
            win = self.game.five_in_a_row()
            self.game.make_move(self.game.current_turn, next_action)
            self.cur_node.last_player = self.game.current_turn #did winner or loser play this move for this playthrough
            self.game.next_turn()
            #play next action
            #check win?
            if win == True: #need to also check draw condition
                winner = self.game.current_turn
                self.game.next_turn()
                loser = self.game.current_turn

                self.backup(search_path, winner)

                return
                
            self.cur_node = self.cur_node.get_child(next_action)

        #we are now at a leaf
        #pick a random next action
        next_action = self.game.get_bot_move(names[self.game.current_turn], t="random")
        if next_action is None: #draw reached on previous move
            self.backup(search_path, -1)
            return
        self.game.check_valid_move(next_action)
        self.game.make_move(self.game.current_turn, next_action)
        self.cur_node.last_player = self.game.current_turn #did winner or loser play this move for this playthrough
        self.game.next_turn()
        self.cur_node.expand(self.game.game_state)
        
        #add new child to search path
        search_path.append(self.cur_node.get_child(next_action))

        game_finished = False
        while game_finished != True:
            #play randomly from both players
            next_action = self.game.get_bot_move(names[self.game.current_turn], t="random")
            if next_action is None:
                self.backup(search_path, -1) #draw reached on previous move
                return
            self.game.check_valid_move(next_action)
            win = self.game.five_in_a_row()
            self.game.make_move(self.game.current_turn, next_action)
            self.game.next_turn()

            if win == True:
                game_finished = True
                loser = self.game.current_turn
                winner = self.game.next_turn()
        
        #backup value scores
        self.backup(search_path, winner)