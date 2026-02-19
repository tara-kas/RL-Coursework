from src.Bots.base_bot import BaseBot
from src.game_datatypes import GameState
from src.game_logic import GameLogic
import numpy as np

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
        if self.children is None:
            self.children = {action: child}
        if action not in self.children:
            self.children[action] = child

    def get_child(self, child_index):
        return self.children[child_index]

class PureMCTS(BaseBot):
    def __init__(self):
        super().__init__()
        self.game = GameLogic(15,15)
        self.root = MCTSNode(self.game.game_state)
        self.cur_node = self.root

    def move(self, game_state, **kwargs):
        t = kwargs.get("t", "random")
        c = kwargs.get("c", 1.4)
        moves = self.game.get_valid_moves()
        if len(moves) == 0:
            return None
        if t == "random":
            choice = np.random.randint(0,len(moves))
            return moves[choice]
        else:
            unexpanded = [m for m in moves if m not in self.cur_node.children.keys()]

            if len(unexpanded) > 0:
                return unexpanded[np.random.randint(len(unexpanded))]
            
            ucb = []
            children = [(action, child) for action, child in self.cur_node.children.items()] #get states into a list
            parent_visits = self.cur_node.visits

            for action, child in children:
                exploitation = child.get_value()
                exploration = c * np.sqrt(np.log(parent_visits)/child.visits)
                ucb.append(exploitation + exploration)

            choice = np.argmax(ucb)
            return children[choice][0]
        
    def backup(self, search_path, winner):
        for node in search_path:
            node.visits += 1
            if node.last_player == winner:
                node.value_sum += 1

    def run(self):
        self.game = GameLogic(15,15)
        
        #override the bots to be the same player
        self.game.bots["mcts"] = self
        self.game.bots["mcts mirror"] = self

        names = ["mcts", "mcts mirror"]
        winner = -1

        self.cur_node = self.root
        search_path = []

        #while we have two bots playing each other, both of them are the same
        while self.cur_node is not None and self.cur_node.children is not None: #while not at leaf
            search_path.append(self.cur_node)
            next_action = self.game.get_bot_move(names[self.game.current_turn], t="ucb")
            if next_action is None:
                self.backup(search_path, -1) #draw reached on last move
                return
            
            x,y = next_action
            win = self.game.five_in_a_row(x,y,self.game.current_turn)
            self.game.make_move(self.game.current_turn, next_action)
            self.cur_node.last_player = self.game.current_turn #did winner or loser play this move for this playthrough
            self.game.next_turn()
            #play next action
            #check win?
            if win == True: #need to also check draw condition
                winner = self.game.current_turn

                self.backup(search_path, winner)

                return
                
            self.cur_node = self.cur_node.get_child(next_action)
        #we are now at a leaf
        #pick a random next action
        next_action = self.game.get_bot_move(names[self.game.current_turn], t="random")
        if next_action is None: #draw reached on previous move
            self.backup(search_path, -1)
            return
        
        self.game.make_move(self.game.current_turn, next_action)
        self.cur_node.last_player = self.game.current_turn #did winner or loser play this move for this playthrough
        self.game.next_turn()
        self.cur_node.expand(MCTSNode(self.game.game_state), next_action)
        
        #add new child to search path
        search_path.append(self.cur_node.get_child(next_action))

        game_finished = False
        while game_finished != True:
            #play randomly from both players
            next_action = self.game.get_bot_move(names[self.game.current_turn], t="random")
            if next_action is None:
                self.backup(search_path, -1) #draw reached on previous move
                return
            x,y = next_action
            win = self.game.five_in_a_row(x,y,self.game.current_turn)
            self.game.make_move(self.game.current_turn, next_action)
            self.game.next_turn()

            if win == True:
                game_finished = True
                winner = self.game.next_turn()
        
        #backup value scores
        self.backup(search_path, winner)

    def pprint(self, node, depth=0, max_depth=3, action=None, prefix="", is_last=True):
        """
        Pretty prints the MCTS tree.
        :param node: The current MCTSNode.
        :param depth: Current depth in the tree.
        :param max_depth: Maximum depth to print to avoid terminal spam.
        :param action: The action (x, y) that led to this node.
        :param prefix: The string prefix for formatting the tree branches.
        :param is_last: Boolean indicating if this is the last child in the branch.
        """
        # Stop if we've reached the maximum requested depth
        if depth > max_depth:
            return
            
        # Safely calculate value (avoid division by zero if visits == 0)
        visits = node.visits
        val = node.get_value() if visits > 0 else 0.0
        
        # Format the display string for the action
        action_str = f"Move {action}" if action is not None else "Root"
        
        # Draw the tree branch
        marker = "└── " if is_last else "├── "
        print(f"{prefix}{marker}{action_str} [Value: {val:.3f} | Visits: {visits}]")
        
        # Recursively print children if the node has been expanded
        if node.children is not None:
            # Prepare the prefix for the next depth level
            new_prefix = prefix + ("    " if is_last else "│   ")
            
            # Convert items to a list so we can identify the last child
            children_items = list(node.children.items())
            for i, (next_action, child_node) in enumerate(children_items):
                is_last_child = (i == len(children_items) - 1)
                self.pprint(
                    child_node, 
                    depth=depth + 1, 
                    max_depth=max_depth, 
                    action=next_action, 
                    prefix=new_prefix, 
                    is_last=is_last_child
                )

Bot = PureMCTS