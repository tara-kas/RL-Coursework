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


def _select_leaf(
    root: MCTSNode,
    c_puct: float,
) -> tuple[MCTSNode, list[tuple[MCTSNode, tuple[int, int] | None]], bool]:
    """
    Run selection from root until an unexpanded node or terminal. Returns (node, path, is_terminal).
    If is_terminal, node is a terminal state (board full or no legal moves); backup with 0, no NN.
    """
    node = root
    path: list[tuple[MCTSNode, tuple[int, int] | None]] = [(root, None)]

    while node.is_expanded() and not is_board_full(node.board):
        legal = get_legal_moves(node.board)
        if not legal:
            return node, path, True
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
            return node, path, True
        node = node.children[best_move]
        path.append((node, best_move))

    if is_board_full(node.board):
        return node, path, True
    legal = get_legal_moves(node.board)
    if not legal:
        return node, path, True
    return node, path, False


def _backup_path(
    path: list[tuple[MCTSNode, tuple[int, int] | None]],
    value: float,
) -> None:
    backup_value = value
    for i in range(len(path) - 1, -1, -1):
        path_node = path[i][0]
        path_node.N += 1
        path_node.W += backup_value
        backup_value = -backup_value


def _run_mcts_simulations(
    root: MCTSNode,
    legal_moves: list[tuple[int, int]],
    model: nn.Module,
    board_size: int,
    num_simulations: int,
    c_puct: float,
    device: torch.device,
    batch_size: int = 32,
) -> None:
    """Run num_simulations MCTS iterations from root in batches. Mutates root and its tree."""
    planes_batch = torch.empty(
        batch_size, 3, board_size, board_size, device=device, dtype=torch.float32
    )
    mask_batch = torch.zeros(
        batch_size, board_size * board_size, device=device, dtype=torch.float32
    )
    n_done = 0
    while n_done < num_simulations:
        current_batch_size = min(batch_size, num_simulations - n_done)
        leaves: list[tuple[MCTSNode, list[tuple[MCTSNode, tuple[int, int] | None]]]] = []
        terminals: list[list[tuple[MCTSNode, tuple[int, int] | None]]] = []

        for _ in range(current_batch_size):
            node, path, is_terminal = _select_leaf(root, c_puct)
            if is_terminal:
                terminals.append(path)
            else:
                leaves.append((node, path))

        for path in terminals:
            _backup_path(path, 0.0)
        n_done += current_batch_size

        if not leaves:
            continue

        unique_nodes = list(dict.fromkeys(node for node, _ in leaves))
        n_unique = len(unique_nodes)

        for u, node in enumerate(unique_nodes):
            planes = preprocess_board(node.board, node.current_player)
            planes_batch[u].copy_(torch.from_numpy(planes))
        mask_batch.zero_()
        for u, node in enumerate(unique_nodes):
            legal = get_legal_moves(node.board)
            for i, j in legal:
                idx = i * board_size + j
                mask_batch[u, idx] = 1.0

        with torch.inference_mode():
            policy_batch, value_batch = model(planes_batch[:n_unique], mask_batch[:n_unique])

        value_batch = value_batch.cpu().numpy().ravel()
        policy_batch = policy_batch.cpu().numpy()

        node_to_idx = {id(n): i for i, n in enumerate(unique_nodes)}
        expanded: set[int] = set()

        for node, path in leaves:
            idx = node_to_idx[id(node)]
            value = float(value_batch[idx])
            policy = policy_batch[idx]
            legal = get_legal_moves(node.board)
            if id(node) not in expanded:
                expanded.add(id(node))
                for move in legal:
                    i, j = move
                    node.P[move] = float(policy[i * board_size + j])
                    child_board = apply_move(node.board, move, node.current_player)
                    opponent = 1 - node.current_player
                    node.children[move] = MCTSNode(child_board, opponent, parent=node)
            _backup_path(path, value)


def run_mcts(board: np.ndarray,
    current_player: int,
    model: nn.Module,
    board_size: int,
    num_simulations: int,
    batch_size: int = 32,
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

    _run_mcts_simulations(root, legal_moves, model, board_size, num_simulations, c_puct, device, batch_size)

    best_move = max(legal_moves, key=lambda m: root.children[m].N if m in root.children else 0)

    return best_move


def run_mcts_with_policy(
    board: np.ndarray,
    current_player: int,
    model: nn.Module,
    board_size: int,
    num_simulations: int,
    batch_size: int = 32,
    c_puct: float = 1.5,
    device: torch.device | None = None,
) -> tuple[tuple[int, int], np.ndarray]:
    """
    Run MCTS and return the best move and the root visit distribution (policy target).
    Returns (best_move, policy) where policy is shape (board_size**2,) with probabilities
    on legal moves summing to 1.
    """
    if device is None:
        device = next(model.parameters()).device

    root = MCTSNode(board, current_player)
    legal_moves = get_legal_moves(board)

    if not legal_moves:
        raise ValueError("No legal moves")

    if len(legal_moves) == 1:
        policy = np.zeros(board_size * board_size, dtype=np.float32)
        i, j = legal_moves[0]
        policy[i * board_size + j] = 1.0
        return legal_moves[0], policy

    _run_mcts_simulations(root, legal_moves, model, board_size, num_simulations, c_puct, device, batch_size)

    total_visits = sum(root.children[m].N for m in legal_moves if m in root.children)
    if total_visits == 0:
        total_visits = 1

    policy = np.zeros(board_size * board_size, dtype=np.float32)
    for move in legal_moves:
        if move in root.children:
            i, j = move
            policy[i * board_size + j] = root.children[move].N / total_visits

    best_move = max(legal_moves, key=lambda m: root.children[m].N if m in root.children else 0)

    return best_move, policy
