import numpy as np
import torch
import torch.nn as nn

from src.gomoku_game import (
    GAME_NOT_OVER,
    WIN,
    apply_move,
    get_game_result,
    get_legal_moves,
    is_board_full,
)
from src.gomoku_utils import preprocess_board


def _flat_move_to_tuple(move_idx: int, board_size: int) -> tuple[int, int]:
    return divmod(move_idx, board_size)


class MCTSNode:
    __slots__ = (
        "board",
        "current_player",
        "parent",
        "last_move",
        "last_player",
        "N",
        "W",
        "legal_moves",
        "legal_moves_idx",
        "priors",
        "child_N",
        "child_W",
        "children",
        "terminal_value",
        "terminal_checked",
    )

    def __init__(
        self,
        board: np.ndarray,
        current_player: int,
        parent: "MCTSNode | None" = None,
        last_move: tuple[int, int] | None = None,
        last_player: int | None = None,
    ) -> None:
        self.board = np.asarray(board, dtype=np.int8)
        self.current_player = current_player
        self.parent = parent
        self.last_move = last_move
        self.last_player = last_player
        self.N = 0
        self.W = 0.0
        self.legal_moves: list[tuple[int, int]] | None = None
        self.legal_moves_idx: np.ndarray | None = None
        self.priors: np.ndarray | None = None
        self.child_N: np.ndarray | None = None
        self.child_W: np.ndarray | None = None
        self.children: list["MCTSNode | None"] | None = None
        self.terminal_value: float | None = None
        self.terminal_checked = False

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def is_expanded(self) -> bool:
        return self.legal_moves_idx is not None


def _get_legal_moves_cached(node: MCTSNode) -> list[tuple[int, int]]:
    if node.legal_moves is None:
        node.legal_moves = get_legal_moves(node.board)
    return node.legal_moves


def _terminal_value(node: MCTSNode) -> float | None:
    """
    Return terminal value from node.current_player perspective if terminal:
      - +1 side-to-move won
      - -1 side-to-move lost
      -  0 draw
    Return None if non-terminal.
    """
    if node.terminal_checked:
        return node.terminal_value

    value: float | None = None
    if node.last_move is not None and node.last_player is not None:
        result = get_game_result(node.board, node.last_move, node.last_player)
        if result != GAME_NOT_OVER:
            if result == WIN:
                value = -1.0
            else:
                value = 0.0
    elif is_board_full(node.board):
        value = 0.0

    if value is None:
        legal = _get_legal_moves_cached(node)
        if not legal:
            value = 0.0

    node.terminal_checked = True
    node.terminal_value = value
    return value


def _expand_node(
    node: MCTSNode,
    policy: np.ndarray,
    board_size: int,
    add_root_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
) -> None:
    legal = _get_legal_moves_cached(node)
    if not legal:
        node.legal_moves = []
        node.legal_moves_idx = np.empty(0, dtype=np.int16)
        node.priors = np.empty(0, dtype=np.float32)
        node.child_N = np.empty(0, dtype=np.int32)
        node.child_W = np.empty(0, dtype=np.float32)
        node.children = []
        return

    legal_idx = np.fromiter(
        (i * board_size + j for i, j in legal),
        count=len(legal),
        dtype=np.int16,
    )
    priors = policy[legal_idx].astype(np.float32, copy=True)
    prior_sum = float(priors.sum())
    if prior_sum <= 0.0:
        priors.fill(1.0 / len(priors))
    else:
        priors /= prior_sum

    if add_root_noise and node.parent is None and len(priors) > 0:
        noise = np.random.dirichlet(
            np.full(len(priors), dirichlet_alpha, dtype=np.float64)
        ).astype(np.float32)
        priors = (1.0 - dirichlet_epsilon) * priors + dirichlet_epsilon * noise

    node.legal_moves_idx = legal_idx
    node.priors = priors
    node.child_N = np.zeros(len(legal_idx), dtype=np.int32)
    node.child_W = np.zeros(len(legal_idx), dtype=np.float32)
    node.children = [None] * len(legal_idx)


def _select_leaf(
    root: MCTSNode,
    c_puct: float,
) -> tuple[MCTSNode, list[tuple[MCTSNode, int | None]], float | None]:
    """
    Run selection from root until an unexpanded node or terminal.
    Returns (node, path, terminal_value). If terminal_value is not None, skip NN and backup directly.
    """
    node = root
    path: list[tuple[MCTSNode, int | None]] = [(root, None)]

    while node.is_expanded():
        tv = _terminal_value(node)
        if tv is not None:
            return node, path, tv
        assert node.priors is not None
        assert node.child_N is not None
        assert node.child_W is not None
        assert node.children is not None
        assert node.legal_moves_idx is not None

        if len(node.legal_moves_idx) == 0:
            return node, path, 0.0

        total_N = float(node.child_N.sum())
        q_values = np.divide(
            node.child_W,
            node.child_N,
            out=np.zeros_like(node.child_W),
            where=node.child_N > 0,
        )
        u_values = c_puct * node.priors * np.sqrt(total_N + 1e-8) / (1.0 + node.child_N)
        best_child_idx = int(np.argmax(q_values + u_values))
        child = node.children[best_child_idx]
        if child is None:
            move = _flat_move_to_tuple(int(node.legal_moves_idx[best_child_idx]), node.board.shape[0])
            child = MCTSNode(
                apply_move(node.board, move, node.current_player),
                1 - node.current_player,
                parent=node,
                last_move=move,
                last_player=node.current_player,
            )
            node.children[best_child_idx] = child
        node = child
        path.append((node, best_child_idx))

    tv = _terminal_value(node)
    if tv is not None:
        return node, path, tv
    return node, path, None


def _backup_path(
    path: list[tuple[MCTSNode, int | None]],
    value: float,
) -> None:
    backup_value = value
    for i in range(len(path) - 1, -1, -1):
        path_node, edge_idx = path[i]
        path_node.N += 1
        path_node.W += backup_value
        if i > 0 and edge_idx is not None:
            parent = path[i - 1][0]
            assert parent.child_N is not None
            assert parent.child_W is not None
            parent.child_N[edge_idx] += 1
            parent.child_W[edge_idx] += backup_value
        backup_value = -backup_value


def _run_mcts_simulations(
    root: MCTSNode,
    model: nn.Module,
    board_size: int,
    num_simulations: int,
    c_puct: float,
    device: torch.device,
    batch_size: int = 32,
    use_amp: bool = False,
    add_root_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
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
        leaves: list[tuple[MCTSNode, list[tuple[MCTSNode, int | None]]]] = []
        terminals: list[tuple[list[tuple[MCTSNode, int | None]], float]] = []

        for _ in range(current_batch_size):
            node, path, terminal_value = _select_leaf(root, c_puct)
            if terminal_value is not None:
                terminals.append((path, terminal_value))
            else:
                leaves.append((node, path))

        for path, value in terminals:
            _backup_path(path, value)
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
            legal = _get_legal_moves_cached(node)
            if legal:
                legal_idx = [i * board_size + j for i, j in legal]
                mask_batch[u, legal_idx] = 1.0

        with torch.inference_mode():
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    policy_batch, value_batch = model(planes_batch[:n_unique], mask_batch[:n_unique])
            else:
                policy_batch, value_batch = model(planes_batch[:n_unique], mask_batch[:n_unique])

        value_batch = value_batch.cpu().numpy().ravel()
        policy_batch = policy_batch.cpu().numpy()

        node_to_idx = {id(n): i for i, n in enumerate(unique_nodes)}
        expanded: set[int] = set()

        for node, path in leaves:
            idx = node_to_idx[id(node)]
            value = float(value_batch[idx])
            policy = policy_batch[idx]
            if id(node) not in expanded:
                expanded.add(id(node))
                _expand_node(
                    node,
                    policy,
                    board_size,
                    add_root_noise=add_root_noise,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_epsilon=dirichlet_epsilon,
                )
            _backup_path(path, value)


def run_mcts(
    board: np.ndarray,
    current_player: int,
    model: nn.Module,
    board_size: int,
    num_simulations: int,
    batch_size: int = 32,
    c_puct: float = 1.5,
    device: torch.device | None = None,
    use_amp: bool = False,
    root: MCTSNode | None = None,
) -> tuple[int, int]:
    """
    Run MCTS from the given state and return the best move.
    """
    if device is None:
        device = next(model.parameters()).device

    if root is None or root.current_player != current_player or not np.array_equal(root.board, board):
        root = MCTSNode(board, current_player)
    legal_moves = _get_legal_moves_cached(root)

    if not legal_moves:
        raise ValueError("No legal moves")

    if len(legal_moves) == 1:
        return legal_moves[0]

    _run_mcts_simulations(root, model, board_size, num_simulations, c_puct, device, batch_size, use_amp)

    assert root.legal_moves_idx is not None
    assert root.child_N is not None
    best_idx = int(np.argmax(root.child_N))
    best_move = _flat_move_to_tuple(int(root.legal_moves_idx[best_idx]), board_size)

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
    temperature: float = 0.0,
    use_amp: bool = False,
    add_root_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    root: MCTSNode | None = None,
) -> tuple[tuple[int, int], np.ndarray, MCTSNode | None]:
    """
    Run MCTS and return the best move and the root visit distribution (policy target).
    Returns (chosen_move, policy) where policy is shape (board_size**2,) with probabilities
    on legal moves summing to 1. If temperature > 0, chosen_move is sampled from
    N^(1/temperature); otherwise argmax over visit counts.
    """
    if device is None:
        device = next(model.parameters()).device

    if root is None or root.current_player != current_player or not np.array_equal(root.board, board):
        root = MCTSNode(board, current_player)
    legal_moves = _get_legal_moves_cached(root)

    if not legal_moves:
        raise ValueError("No legal moves")

    if len(legal_moves) == 1:
        policy = np.zeros(board_size * board_size, dtype=np.float32)
        i, j = legal_moves[0]
        policy[i * board_size + j] = 1.0
        return legal_moves[0], policy, None

    _run_mcts_simulations(
        root,
        model,
        board_size,
        num_simulations,
        c_puct,
        device,
        batch_size,
        use_amp,
        add_root_noise=add_root_noise,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )

    assert root.legal_moves_idx is not None
    assert root.child_N is not None
    assert root.children is not None

    total_visits = int(root.child_N.sum())
    if total_visits == 0:
        total_visits = 1

    policy = np.zeros(board_size * board_size, dtype=np.float32)
    for idx, move_idx in enumerate(root.legal_moves_idx):
        policy[int(move_idx)] = root.child_N[idx] / total_visits

    if temperature <= 0.0:
        chosen_idx = int(np.argmax(root.child_N))
    else:
        visits = root.child_N.astype(np.float64, copy=False)
        probs = np.power(visits + 1e-8, 1.0 / temperature)
        probs /= probs.sum()
        chosen_idx = int(np.random.choice(len(root.legal_moves_idx), p=probs))

    chosen_move = _flat_move_to_tuple(int(root.legal_moves_idx[chosen_idx]), board_size)
    next_root = root.children[chosen_idx]
    if next_root is not None:
        next_root.parent = None

    return chosen_move, policy, next_root
