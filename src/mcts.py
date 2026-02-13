import numpy as np
import torch

from src.gomoku_game import apply_move, get_legal_moves, is_board_full
from src.gomoku_utils import preprocess_board
import torch.nn as nn


class MCTSNode():
    def __init__(self, board: np.ndarray, current_player: int, parent: "MCTSNode | None" = None):
        self.board = board
        self.current_player = current_player
        self.parent = parent
        self.children: dict[tuple[int, int], MCTSNode] = {}
        self.N: int = 0
        self.W: float = 0.0  # Total value from this node's player perspective
        self.P: dict[tuple[int, int], float] = {}

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0


def run_mcts(board: np.ndarray,
    current_player: int,
    model: nn.Module,
    board_size: int,
    num_simulations: int,
    c_puct: float = 1.5,
    device: torch.device | None = None,
) -> tuple[int, int]:
    """
    Run MCTS from the given state and return the best move.
    """
    if device is None:
        device = next(model.parameters()).device

    root = MCTSNode(board, current_player)
    legal_moves = get_legal_moves(board)

    if not legal_moves:
        raise ValueError("No legal moves")

    if len(legal_moves) == 1:
        return legal_moves[0]

    for _ in range(num_simulations):
        node = root
        path: list[tuple[MCTSNode, tuple[int, int] | None]] = [(root, None)]

        # Select the best move until a leaf node is reached
        while node.is_expanded() and not is_board_full(node.board):
            legal = get_legal_moves(node.board)

            if not legal:
                break

            total_N = sum(node.children[m].N for m in legal)
            best_score = -float("inf")
            best_move = None

            for move in legal:
                child = node.children[move]

                p = node.P.get(move, 1.0 / len(legal))

                u = c_puct * p * np.sqrt(total_N + 1e-8) / (1 + child.N)

                score = child.Q + u

                if score > best_score:
                    best_score = score
                    best_move = move

            if best_move is None:
                break

            node = node.children[best_move]
            path.append((node, best_move))

        if is_board_full(node.board):
            backup_value = 0.0

        legal = get_legal_moves(node.board)

        if not legal:
            backup_value = 0.0
        else:
            # Evaluate the best move
            planes = preprocess_board(node.board, node.current_player)
            x = torch.tensor(planes, dtype=torch.float32).unsqueeze(0).to(device)

            mask = torch.zeros(1, board_size * board_size, device=device)

            for i, j in legal:
                idx = i * board_size + j
                mask[0, idx] = 1.0

            with torch.no_grad():
                policy, value = model(x, mask)

            value = value.item()
            policy = policy[0].cpu().numpy()

            for move in legal:
                i, j = move
                idx = i * board_size + j

                node.P[move] = float(policy[idx])

                child_board = apply_move(node.board, move, node.current_player)

                opponent = 1 - node.current_player
                node.children[move] = MCTSNode(child_board, opponent, parent=node)

            backup_value = value

        # Backup the value from the leaf's current_player perspective
        for i in range(len(path) - 1, -1, -1):
            # Alternating players as each player is the opposite of the previous player
            path_node = path[i][0]
            path_node.N += 1
            path_node.W += backup_value
            backup_value = -backup_value

    # Return the move with the highest N (number of visits)
    best_move = max(legal_moves, key=lambda m: root.children[m].N if m in root.children else 0)

    return best_move