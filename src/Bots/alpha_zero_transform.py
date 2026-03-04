import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Bots.base_bot import BaseBot
from src.game_datatypes import GameState
from src.gomoku_game import get_legal_moves
from src.mcts import run_mcts, run_mcts_with_policy
from src.model_loader import DEFAULT_WEIGHTS_PATH, load_weights, save_weights as save_weights_to_path


def _maybe_compile(model: nn.Module, enable: bool = True) -> nn.Module:
    """Compile model with torch.compile if PyTorch >= 2 and enable is True. First run may be slow (tracing)."""
    if not enable:
        return model
    # torch.compile with inductor requires Triton, which is not supported on Windows
    import sys
    if sys.platform == "win32":
        return model
    try:
        if hasattr(torch, "compile") and tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 0):
            return torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass
    return model



def _create_2d_sinusoidal_encoding(board_size: int, embed_dim: int) -> torch.Tensor:
    """Create 2D sinusoidal positional encoding. Returns (seq_len, embed_dim)."""
    d_half = embed_dim // 2
    div_term = torch.exp(torch.arange(0, d_half, 2).float() * (-np.log(10000.0) / d_half))

    pe = torch.zeros(board_size * board_size, embed_dim)

    for i in range(board_size):
        for j in range(board_size):
            pos = i * board_size + j

            # Row (i) encoding -> first half
            pe[pos, 0:d_half:2] = torch.sin(torch.tensor(i, dtype=torch.float32) * div_term)
            pe[pos, 1:d_half:2] = torch.cos(torch.tensor(i, dtype=torch.float32) * div_term)

            # Col (j) encoding -> second half
            pe[pos, d_half::2] = torch.sin(torch.tensor(j, dtype=torch.float32) * div_term)
            pe[pos, d_half + 1::2] = torch.cos(torch.tensor(j, dtype=torch.float32) * div_term)

    return pe


class Bot(BaseBot):
    def __init__(
        self,
        num_simulations: int = 100,
        board_size: int = 15,
        device: str | torch.device | None = None,
        weights_path: str | None = None,
        model: nn.Module | None = None,
        compile_model: bool = True,
        mcts_batch_size: int = 32,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.num_simulations = num_simulations
        self.board_size = board_size
        self.mcts_batch_size = mcts_batch_size

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = AlphaZeroTransform(board_size=board_size).to(self.device)
            path = weights_path if weights_path is not None else DEFAULT_WEIGHTS_PATH
            if os.path.isfile(path):
                load_weights(self.model, path, self.device)
        self.model.eval()
        self.model = _maybe_compile(self.model, compile_model)

    def save_weights(self, path: str) -> None:
        """Save the current model state dict to path."""
        save_weights_to_path(self.model, path)

    def predict(
        self,
        board_state: np.ndarray,
        current_player: int | None = None,
    ) -> tuple[int, int]:
        """
        Predict the next move from a raw board state. Reuses this bot's model (no reinit).
        board_state: (H, W) array, -1 empty, 0 and 1 for the two players.
        current_player: who is to move (0 or 1). If None, inferred from stone counts.
        Returns (x, y) the chosen move.
        """
        if current_player is None:
            n0 = int(np.sum(board_state == 0))
            n1 = int(np.sum(board_state == 1))
            current_player = 0 if n0 == n1 else 1
        move = run_mcts(
            board_state.copy(),
            current_player,
            self.model,
            self.board_size,
            num_simulations=self.num_simulations,
            batch_size=self.mcts_batch_size,
            c_puct=1.5,
            device=self.device,
        )
        return move

    def get_move_and_policy(
        self,
        board: np.ndarray,
        current_player: int,
        c_puct: float = 1.5,
    ) -> tuple[tuple[int, int], np.ndarray]:
        """
        Run MCTS and return (move, policy) for self-play training. Uses this bot's model.
        """
        return run_mcts_with_policy(
            board,
            current_player,
            self.model,
            self.board_size,
            num_simulations=self.num_simulations,
            batch_size=self.mcts_batch_size,
            c_puct=c_puct,
            device=self.device,
        )

    def move(self, game_state: GameState) -> tuple[int, int] | None:
        legal_moves = get_legal_moves(game_state.board)

        if not legal_moves:
            return None

        current_player = game_state.current_player if game_state.current_player is not None else 0

        return run_mcts(
            game_state.board.copy(),
            current_player,
            self.model,
            self.board_size,
            num_simulations=self.num_simulations,
            batch_size=self.mcts_batch_size,
            c_puct=1.5,
            device=self.device,
        )


class AlphaZeroTransform(nn.Module):
    def __init__(self, board_size=15, embed_dim=128, n_heads=8, n_layers=6):
        super().__init__()

        self.board_size = board_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.num_tokens = board_size ** 2

        # Maps the 3 planes to the embedding space
        self.input_projection = nn.Linear(3, embed_dim)

        # 2D sinusoidal positional encoding
        pos_enc = _create_2d_sinusoidal_encoding(board_size, embed_dim)
        self.register_buffer("positional_embedding", pos_enc.unsqueeze(0))

        # Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Policy Head (Actor): Outputs logits for 225 moves
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Flatten()
        )

        # Value Head (Critic): Mean-pooled features -> scalar (-1 to 1)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, 3, H, W)
        mask: optional (batch, 225) where 1=legal, 0=illegal. If None, no masking.
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, 3, -1).transpose(1, 2)

        # Project to embedding space and add 2D positional encoding
        x = self.input_projection(x) + self.positional_embedding

        # Run through Transformer
        features = self.transformer(x)

        # Policy: logits for each position
        logits = self.policy_head(features)

        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)

        policy = F.softmax(logits, dim=-1)

        # Value: mean pool over tokens
        pooled = features.mean(dim=1)
        value = self.value_head(pooled)

        return policy, value


def predict(
    board_state: np.ndarray,
    current_player: int | None = None,
    weights_path: str | None = None,
    bot: Bot | None = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """
    Standardized API for cross-group testing: predict the next move from a board state.
    For repeated predictions, create one Bot and pass it as bot=... or call bot.predict() directly.
    board_state: (H, W) array, -1 empty, 0 and 1 for the two players.
    current_player: who is to move (0 or 1). If None, inferred from stone counts.
    weights_path: optional path to model weights (ignored if bot is provided).
    bot: optional pre-created Bot instance; if given, uses it instead of creating one.
    **kwargs: passed to Bot when bot is not provided (e.g. num_simulations, board_size).
    Returns (x, y) the chosen move.
    """
    if bot is not None:
        return bot.predict(board_state, current_player=current_player)
    b = Bot(weights_path=weights_path, **kwargs)
    return b.predict(board_state, current_player=current_player)
