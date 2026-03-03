from src.Bots.base_bot import BaseBot
from src.game_datatypes import GameState
from src.game_logic import GameLogic
import numpy as np
import pickle

#TODO: logic for choosing moves against player
# heuristic based/optimised non pure MCTS
# export function to save a trained tree

class MCTSNode:
    def __init__(self, state: GameState):
        self.visits = 0
        self.value_sum = 0
        self.children = None
        self.state = state
        self.last_player = -1
        
        ### NEW: RAVE stats
        self.rave_visits = 0
        self.rave_value_sum = 0

    def get_value(self):
        return self.value_sum / self.visits

    def expand(self, child: 'MCTSNode', action):
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

    def get_node_ucbs(self, c, rave_k=300):
        ### NEW: Blending Standard MCTS and RAVE
        ucb = []
        children = [(action, child) for action, child in self.cur_node.children.items()]
        parent_visits = max(1, self.cur_node.visits)

        for action, child in children:
            if child.visits == 0 and child.rave_visits == 0:
                ucb.append(float('inf'))
                continue

            # Standard exploitation
            exploitation = child.get_value() if child.visits > 0 else 0.5
            
            # RAVE estimation
            rave_val = child.rave_value_sum / child.rave_visits if child.rave_visits > 0 else 0.5
            
            # Blending factor Beta: high RAVE reliance at low visits
            beta = rave_k / (rave_k + child.visits)
            blended_val = (1 - beta) * exploitation + beta * rave_val
            
            # Exploration
            exploration = c * np.sqrt(np.log(parent_visits) / max(1, child.visits))
            
            ucb.append(blended_val + exploration)
            
        return children, ucb

    def move(self, game_state, **kwargs):
        t = kwargs.get("t", "random")
        c = kwargs.get("c", 1.4)
        c_pw = kwargs.get("c_pw", 1.5)  # Progressive Widening Constant
        alpha = kwargs.get("alpha", 0.5) # Progressive Widening Power
        
        moves = self.game.get_valid_moves()
        if len(moves) == 0:
            return None
            
        if t == "random":
            print("random")
            choice = np.random.randint(0,len(moves))
            return moves[choice]
        else:
            if self.cur_node.children is None:
                self.cur_node.children = {}
                
            unexpanded = [m for m in moves if m not in self.cur_node.children.keys()]

            ### NEW: Progressive Widening Logic
            parent_visits = max(1, self.cur_node.visits)
            k = int(np.ceil(c_pw * (parent_visits ** alpha)))

            # Only expand a new node if we haven't hit our progressive width limit
            if len(self.cur_node.children) < k and len(unexpanded) > 0:
                return unexpanded[np.random.randint(0,len(unexpanded))]
            
            if len(self.cur_node.children) == 0:
                # Failsafe
                return moves[np.random.randint(0,len(moves))]

            # Force UCB selection among the limited pool of allowed children
            children, ucb = self.get_node_ucbs(c)
            choice = np.argmax(ucb)
            return children[choice][0]
        
    def backup(self, search_path, winner, played_moves):
        ### NEW: Updating standard AND RAVE stats
        for node in search_path:
            node.visits += 1
            if node.last_player == winner:
                node.value_sum += 1

            # RAVE update: Did the player who owns this child branch play this move 
            # anywhere during the simulation? If yes, update RAVE stats.
            if node.children is not None:
                for action, child in node.children.items():
                    mover = child.last_player
                    if mover in played_moves and action in played_moves[mover]:
                        child.rave_visits += 1
                        if mover == winner:
                            child.rave_value_sum += 1

    def run(self):
        self.game = GameLogic(15,15)
        self.game.bots["mcts"] = self
        self.game.bots["mcts mirror"] = self

        names = ["mcts", "mcts mirror"]
        winner = -1

        self.cur_node = self.root
        search_path = []
        
        ### NEW: Track all moves played by each player this iteration for RAVE
        played_moves = {0: set(), 1: set()} 

        while self.cur_node is not None: 
            search_path.append(self.cur_node)
            next_action = self.game.get_bot_move(names[self.game.current_turn], t="ucb")
            
            if next_action is None:
                self.backup(search_path, -1, played_moves)
                return
                
            ### NEW: Record tree-phase action
            played_moves[self.game.current_turn].add(next_action)
            
            if self.cur_node.children is None or next_action not in self.cur_node.children.keys():
                self.cur_node.expand(MCTSNode(self.game.game_state), next_action)
                self.cur_node = self.cur_node.get_child(next_action)
                search_path.append(self.cur_node)
                
                x,y = next_action
                win = self.game.five_in_a_row(x,y,self.game.current_turn)
                self.cur_node.last_player = self.game.current_turn 
                self.game.make_move(self.game.current_turn, next_action)
                self.game.next_turn()
                
                if win: 
                    winner = self.cur_node.last_player
                    self.backup(search_path, winner, played_moves)
                    return
                break

            self.cur_node = self.cur_node.get_child(next_action)
            
            x,y = next_action
            win = self.game.five_in_a_row(x,y,self.game.current_turn)
            self.cur_node.last_player = self.game.current_turn 
            self.game.make_move(self.game.current_turn, next_action)
            self.game.next_turn()
            
            if win: 
                winner = self.cur_node.last_player
                self.backup(search_path, winner, played_moves)
                return

        game_finished = False
        while not game_finished:
            next_action = self.game.get_bot_move(names[self.game.current_turn], t="random")
            if next_action is None:
                self.backup(search_path, -1, played_moves) 
                return
                
            ### NEW: Record rollout-phase action
            played_moves[self.game.current_turn].add(next_action)
            
            x,y = next_action
            win = self.game.five_in_a_row(x,y,self.game.current_turn)
            
            # Store who played the move before the turn changes!
            current_mover = self.game.current_turn
            self.game.make_move(self.game.current_turn, next_action)
            self.game.next_turn()

            if win:
                game_finished = True
                winner = current_mover
        
        self.backup(search_path, winner, played_moves)

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

    def save_tree(self, filename="mcts_tree.pkl"):
        """
        Exports the current MCTS root node and all its children to a binary file.
        """
        try:
            with open(filename, 'wb') as f:
                # We only need to save the root; pickle will automatically traverse 
                # and save the entire dictionary of children connected to it.
                pickle.dump(self.root, f)
            print(f"Successfully exported tree with {self.root.visits} visits to {filename}")
        except Exception as e:
            print(f"Failed to save tree: {e}")

    def load_tree(self, filename="mcts_tree.pkl"):
        """
        Imports an MCTS tree from a binary file and sets it as the active tree.
        """
        try:
            with open(filename, 'rb') as f:
                self.root = pickle.load(f)
                # Reset the current working node back to the newly loaded root
                self.cur_node = self.root
            print(f"Successfully imported tree with {self.root.visits} visits from {filename}")
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
        except Exception as e:
            print(f"Failed to load tree: {e}")

Bot = PureMCTS