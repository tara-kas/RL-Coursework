import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Bots.base_bot import BaseBot
from src.game_datatypes import GameState
from src.gomoku_game import get_legal_moves
from src.mcts import run_mcts

import random



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
    ):
        super().__init__()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = AlphaZeroTransform(board_size=board_size).to(self.device)
        self.model.eval()
        self.num_simulations = num_simulations
        self.board_size = board_size
        self.device = device

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
