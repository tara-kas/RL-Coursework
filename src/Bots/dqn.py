"""
DQN agent for Gomoku: Q-network with replay buffer and epsilon-greedy exploration.
Used for training with TD learning when --agent_type dqn.
"""
import math
import os
import random
from collections import deque
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gomoku_game import (
    GAME_NOT_OVER,
    WIN,
    DRAW,
    apply_move,
    get_game_result,
    get_legal_moves,
)
from src.gomoku_utils import preprocess_board
from src.Bots.base_bot import BaseBot
from src.Bots.heuristic_tactical import predict as heuristic_predict
from src.game_datatypes import GameState
from src.gomoku_game import get_legal_moves as _get_legal_moves
from src.model_loader import load_weights


def move_to_idx(x: int, y: int, board_size: int) -> int:
    """Convert (x, y) move to flattened action index."""
    return x * board_size + y


def idx_to_move(idx: int, board_size: int) -> tuple[int, int]:
    """Convert flattened action index to (x, y) move."""
    return idx // board_size, idx % board_size


class DQN(nn.Module):
    """
    Q-network: takes 3-plane input (same as AlphaZero), outputs Q(s,a) for 225 actions.
    Uses CNN backbone + MLP head for efficiency.
    """

    def __init__(self, board_size: int = 15, channels: int = 128, hidden: int = 256):
        super().__init__()
        self.board_size = board_size
        self.num_actions = board_size * board_size

        # CNN backbone: (batch, 3, H, W) -> (batch, channels, H, W)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, channels, 3, padding=1),
            nn.ReLU(),
        )
        # Per-position features -> Q per position
        self.q_head = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch, 3, H, W)
        mask: optional (batch, 225) where 1=legal, 0=illegal
        Returns: (batch, 225) Q-values
        """
        batch_size = x.shape[0]
        features = self.conv(x)  # (batch, channels, H, W)
        features = features.permute(0, 2, 3, 1)  # (batch, H, W, channels)
        q_values = self.q_head(features).view(batch_size, -1)

        if mask is not None:
            # Use -1e4 instead of -1e9: float16 range is ~±65504; -1e9 causes overflow under AMP
            q_values = q_values.masked_fill(mask == 0, -1e4)

        return q_values


class Bot(BaseBot):
    """DQN bot for play: loads DQN weights and plays greedily (epsilon=0)."""

    def __init__(
        self,
        board_size: int = 15,
        device: str | torch.device | None = None,
        weights_path: str | None = None,
        use_amp: bool = False,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.board_size = board_size
        self.use_amp = use_amp
        self.model = DQN(board_size=board_size).to(self.device)
        if weights_path and os.path.isfile(weights_path):
            load_weights(self.model, weights_path, self.device)
        self.model.eval()

    def move(self, game_state: GameState) -> tuple[int, int] | None:
        legal = _get_legal_moves(game_state.board)
        if not legal:
            return None
        current_player = game_state.current_player if game_state.current_player is not None else 0
        _, move = select_action(
            self.model,
            game_state.board,
            current_player,
            self.board_size,
            self.device,
            epsilon=0.0,
            use_amp=self.use_amp,
        )
        return move


class ReplayBuffer:
    """Fixed-size replay buffer of (state, action_idx, reward, next_state, done) with optional terminal oversampling."""

    def __init__(self, capacity: int, terminal_capacity: int = 50000):
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self.terminal_buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=terminal_capacity
        )

    def push(
        self,
        state: np.ndarray,
        action_idx: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action_idx, reward, next_state, done))
        if done:
            self.terminal_buffer.append((state, action_idx, reward, next_state, done))

    def sample(
        self,
        batch_size: int,
        terminal_fraction: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_terminal = min(
            len(self.terminal_buffer),
            math.ceil(batch_size * terminal_fraction),
        )
        n_rest = batch_size - n_terminal
        batch_tuples: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        if n_terminal > 0 and len(self.terminal_buffer) > 0:
            terminal_sample = random.sample(
                list(self.terminal_buffer),
                min(n_terminal, len(self.terminal_buffer)),
            )
            batch_tuples.extend(terminal_sample)
        need = batch_size - len(batch_tuples)
        if need > 0 and len(self.buffer) > 0:
            main_sample = random.sample(
                list(self.buffer),
                min(need, len(self.buffer)),
            )
            batch_tuples.extend(main_sample)
        if len(batch_tuples) == 0:
            return (
                np.zeros((0, 3, 15, 15), dtype=np.float32),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
                np.zeros((0, 3, 15, 15), dtype=np.float32),
                np.array([], dtype=np.float32),
            )
        random.shuffle(batch_tuples)
        states, actions, rewards, next_states, dones = zip(*batch_tuples)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def get_epsilon(
    step: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
) -> float:
    """Linear decay: epsilon_start -> epsilon_end over epsilon_decay_steps."""
    if step >= epsilon_decay_steps:
        return epsilon_end
    t = step / epsilon_decay_steps
    return epsilon_start + t * (epsilon_end - epsilon_start)


def select_action(
    model: nn.Module,
    board: np.ndarray,
    current_player: int,
    board_size: int,
    device: torch.device,
    epsilon: float,
    use_amp: bool = False,
) -> tuple[int, tuple[int, int]]:
    """
    Epsilon-greedy: with prob epsilon random legal move, else argmax Q.
    Returns (action_idx, (x, y)).
    """
    legal = get_legal_moves(board)
    if not legal:
        raise ValueError("No legal moves")

    if random.random() < epsilon:
        move = random.choice(legal)
        return move_to_idx(move[0], move[1], board_size), move

    state = preprocess_board(board, current_player)
    mask = (board == -1).astype(np.float32).flatten()
    state_t = torch.tensor(
        state[np.newaxis, ...], dtype=torch.float32, device=device
    )
    mask_t = torch.tensor(mask[np.newaxis, ...], dtype=torch.float32, device=device)

    model.eval()
    with torch.inference_mode():
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                q_values = model(state_t, mask_t)
        else:
            q_values = model(state_t, mask_t)
    q_np = q_values.cpu().numpy().ravel()

    best_idx = -1
    best_q = -float("inf")
    for x, y in legal:
        idx = move_to_idx(x, y, board_size)
        if q_np[idx] > best_q:
            best_q = q_np[idx]
            best_idx = idx

    if best_idx < 0:
        move = random.choice(legal)
        return move_to_idx(move[0], move[1], board_size), move

    return best_idx, idx_to_move(best_idx, board_size)


def dqn_self_play(
    model: nn.Module,
    replay_buffer: ReplayBuffer,
    board_size: int,
    num_games: int,
    device: torch.device,
    epsilon: float,
    heuristic_prob: float = 0.0,
    league_model: nn.Module | None = None,
    league_prob: float = 0.0,
    progress_callback: Callable[[int, int], None] | None = None,
    use_amp: bool = False,
) -> tuple[int, int, int, int]:
    """
    Play num_games and push (s, a, r, s', done) transitions to replay_buffer.
    Opponent per game: with probability league_prob use league_model, with heuristic_prob use
    heuristic bot, else self-play (current model). Probabilities are league_prob, heuristic_prob,
    and 1 - league_prob - heuristic_prob.
    Returns (total_steps, wins, losses, draws, games_league, games_heuristic, games_self_play) for player 0.
    """
    model.eval()
    if league_model is not None:
        league_model.eval()
    total_steps = 0
    wins = 0
    losses = 0
    draws = 0
    games_league = 0
    games_heuristic = 0
    games_self_play = 0

    for g in range(num_games):
        if progress_callback is not None:
            progress_callback(g + 1, num_games)

        r = random.random()
        use_league_opponent = league_model is not None and r < league_prob
        use_heuristic_opponent = not use_league_opponent and r < league_prob + heuristic_prob
        if use_league_opponent:
            games_league += 1
        elif use_heuristic_opponent:
            games_heuristic += 1
        else:
            games_self_play += 1
        board = np.full((board_size, board_size), -1, dtype=np.int32)
        current_player = 0

        while True:
            state = preprocess_board(board.copy(), current_player)

            if current_player == 1 and use_league_opponent:
                _, move = select_action(
                    league_model, board, current_player, board_size, device, epsilon=0.0, use_amp=use_amp
                )
                action_idx = move_to_idx(move[0], move[1], board_size)
            elif current_player == 1 and use_heuristic_opponent:
                move = heuristic_predict(board.copy(), current_player)
                action_idx = move_to_idx(move[0], move[1], board_size)
            else:
                action_idx, move = select_action(
                    model, board, current_player, board_size, device, epsilon, use_amp
                )

            board = apply_move(board, move, current_player)
            result = get_game_result(board, move, current_player)

            if result != GAME_NOT_OVER:
                if result == WIN:
                    winner = current_player
                else:
                    winner = -1
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1
                reward = 1.0 if winner == current_player else (-1.0 if winner != -1 else 0.0)
                next_state = preprocess_board(board.copy(), current_player)
                done = True
                replay_buffer.push(state, action_idx, reward, next_state, done)
                total_steps += 1
                break

            next_state = preprocess_board(board.copy(), 1 - current_player)
            done = False
            reward = 0.0
            replay_buffer.push(state, action_idx, reward, next_state, done)
            total_steps += 1
            current_player = 1 - current_player

    return total_steps, wins, losses, draws, games_league, games_heuristic, games_self_play


def evaluate_dqn(
    model: nn.Module,
    board_size: int,
    device: torch.device,
    num_games: int,
    opponent: str,
    use_amp: bool = False,
) -> dict:
    """
    Run num_games with DQN as player 0 vs opponent (player 1). opponent is "random" or "heuristic".
    Returns dict with wins, losses, draws, win_rate, avg_return (mean game return for DQN).
    """
    model.eval()
    wins = 0
    losses = 0
    draws = 0
    for _ in range(num_games):
        board = np.full((board_size, board_size), -1, dtype=np.int32)
        current_player = 0
        while True:
            if current_player == 0:
                _, move = select_action(
                    model, board, current_player, board_size, device, epsilon=0.0, use_amp=use_amp
                )
            else:
                legal = get_legal_moves(board)
                if not legal:
                    break
                if opponent == "random":
                    move = random.choice(legal)
                else:
                    move = heuristic_predict(board.copy(), current_player)
            board = apply_move(board, move, current_player)
            result = get_game_result(board, move, current_player)
            if result != GAME_NOT_OVER:
                if result == WIN:
                    winner = current_player
                else:
                    winner = -1
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1
                break
            current_player = 1 - current_player
    total = wins + losses + draws
    win_rate = wins / total if total else 0.0
    avg_return = (wins - losses) / total if total else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "avg_return": avg_return,
    }


def dqn_train_step(
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    board_size: int,
    device: torch.device,
    gamma: float = 0.99,
    use_amp: bool = False,
) -> float:
    """
    One DQN training step: TD target, MSE loss.
    batch: (states, action_indices, rewards, next_states, dones)
    Returns loss value.
    """
    states, actions, rewards, next_states, dones = batch
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    # next_state mask: plane 2 = empty cells; legal = where empty
    next_masks = (next_states[:, 2, :, :] > 0.5).reshape(len(next_states), -1).astype(
        np.float32
    )
    next_masks_t = torch.tensor(next_masks, dtype=torch.float32, device=device)

    model.train()
    target_model.eval()

    q_pred = model(states_t, None).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        q_next = target_model(next_states_t, next_masks_t)
        q_next_max = q_next.max(dim=1).values
        target = rewards_t + gamma * (1 - dones_t) * q_next_max

    loss = F.mse_loss(q_pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
