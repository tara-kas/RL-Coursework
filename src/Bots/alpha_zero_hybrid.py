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
from src.model_loader import (
    DEFAULT_WEIGHTS_PATH,
    load_weights,
    save_weights as save_weights_to_path,
)


def _maybe_compile(model: nn.Module, enable: bool = True) -> nn.Module:
    """Compile model with torch.compile when available."""
    if not enable:
        return model
    import sys

    if sys.platform == "win32":
        return model
    try:
        if hasattr(torch, "compile") and tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 0):
            return torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass
    return model


class ResBlock(nn.Module):
    """Residual block with optional projection for channel changes."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.proj is None else self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with attention map capture."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, need_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_in = self.norm1(x)
        attn_out, attn_weights = self.attn(
            attn_in,
            attn_in,
            attn_in,
            need_weights=need_attn,
            average_attn_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class AlphaZeroHybrid(nn.Module):
    """
    Hybrid CNN-Transformer AlphaZero backbone for 9x9 Gomoku.

    Stage 1: CNN feature extractor.
    Stage 2: Flatten to 81x256 and add learned 2D positional embeddings.
    Stage 3: 4-layer Transformer encoder.
    Stage 4: Policy head -> distribution over 81 moves.
    Stage 5: Value head -> scalar in [-1, 1].
    """

    def __init__(self, board_size: int = 9, dropout: float = 0.1):
        super().__init__()
        if board_size != 9:
            raise ValueError("AlphaZeroHybrid is configured for 9x9 Gomoku only.")
        self.board_size = board_size
        self.seq_len = board_size * board_size
        self.embed_dim = 256

        # Stage 1: CNN feature extractor
        self.conv_input = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(128)
        self.res_128 = nn.ModuleList([ResBlock(128, 128), ResBlock(128, 128)])
        self.res_256 = nn.ModuleList([ResBlock(128, 256), ResBlock(256, 256)])

        # Stage 2: learned row/col positional embedding, 128 + 128 -> 256
        self.row_embed = nn.Embedding(board_size, 128)
        self.col_embed = nn.Embedding(board_size, 128)

        # Stage 3: Transformer encoder
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim=256, num_heads=8, dropout=dropout) for _ in range(4)]
        )

        # Stage 4: Policy head
        self.policy_conv1 = nn.Conv2d(256, 128, kernel_size=1)
        self.policy_conv2 = nn.Conv2d(128, 1, kernel_size=1)

        # Stage 5: Value head
        self.value_fc1 = nn.Linear(256, 256)
        self.value_fc2 = nn.Linear(256, 128)
        self.value_fc3 = nn.Linear(128, 1)
        self.value_drop = nn.Dropout(dropout)

    def _build_positional_encoding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(self.board_size, device=device)
        cols = torch.arange(self.board_size, device=device)
        row_vec = self.row_embed(rows)  # (9, 128)
        col_vec = self.col_embed(cols)  # (9, 128)
        row_grid = row_vec[:, None, :].expand(self.board_size, self.board_size, -1)
        col_grid = col_vec[None, :, :].expand(self.board_size, self.board_size, -1)
        pos = torch.cat([row_grid, col_grid], dim=-1).reshape(self.seq_len, self.embed_dim)
        return pos.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        # Compatibility shim: the existing MCTS preprocessing uses an "empty" plane.
        # We enforce the requested "color-to-play" channel as a constant 1 plane.
        color_plane = torch.ones_like(x[:, 2:3])
        x = torch.cat([x[:, :2], color_plane], dim=1)

        # Stage 1
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_128:
            x = block(x)
        for block in self.res_256:
            x = block(x)

        # Stage 2
        bsz = x.size(0)
        seq = x.flatten(2).transpose(1, 2)  # (B, 81, 256)
        seq = seq + self._build_positional_encoding(bsz, x.device)

        # Stage 3
        attn_maps: list[torch.Tensor] = []
        for block in self.transformer_blocks:
            seq, attn = block(seq, need_attn=return_attention)
            if return_attention and attn is not None:
                attn_maps.append(attn.detach())

        # Stage 4
        spatial = seq.transpose(1, 2).reshape(bsz, 256, self.board_size, self.board_size)
        p = F.relu(self.policy_conv1(spatial))
        logits = self.policy_conv2(p).flatten(1)  # (B, 81)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e4)
        policy = F.softmax(logits, dim=-1)

        # Stage 5
        pooled = seq.mean(dim=1)  # global avg pool on sequence
        v = F.relu(self.value_fc1(pooled))
        v = self.value_drop(v)
        v = F.relu(self.value_fc2(v))
        v = self.value_drop(v)
        value = torch.tanh(self.value_fc3(v))

        if return_attention:
            return policy, value, attn_maps
        return policy, value


class Bot(BaseBot):
    def __init__(
        self,
        num_simulations: int = 800,
        board_size: int = 9,
        device: str | torch.device | None = None,
        weights_path: str | None = None,
        model: nn.Module | None = None,
        compile_model: bool = True,
        mcts_batch_size: int = 32,
        use_amp: bool = False,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_simulations = num_simulations
        self.board_size = board_size
        self.mcts_batch_size = mcts_batch_size
        self.use_amp = use_amp

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = AlphaZeroHybrid(board_size=board_size).to(self.device)
            path = weights_path if weights_path is not None else DEFAULT_WEIGHTS_PATH
            if os.path.isfile(path):
                load_weights(self.model, path, self.device)
        self.model.eval()
        self.model = _maybe_compile(self.model, compile_model)

    def save_weights(self, path: str) -> None:
        save_weights_to_path(self.model, path)

    def predict(
        self,
        board_state: np.ndarray,
        current_player: int | None = None,
    ) -> tuple[int, int]:
        if current_player is None:
            n0 = int(np.sum(board_state == 0))
            n1 = int(np.sum(board_state == 1))
            current_player = 0 if n0 == n1 else 1
        return run_mcts(
            board_state.copy(),
            current_player,
            self.model,
            self.board_size,
            num_simulations=self.num_simulations,
            batch_size=self.mcts_batch_size,
            c_puct=1.5,
            device=self.device,
            use_amp=self.use_amp,
        )

    def get_move_and_policy(
        self,
        board: np.ndarray,
        current_player: int,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        add_root_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> tuple[tuple[int, int], np.ndarray]:
        return run_mcts_with_policy(
            board,
            current_player,
            self.model,
            self.board_size,
            num_simulations=self.num_simulations,
            batch_size=self.mcts_batch_size,
            c_puct=c_puct,
            device=self.device,
            temperature=temperature,
            use_amp=self.use_amp,
            add_root_noise=add_root_noise,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
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
            use_amp=self.use_amp,
        )


def predict(
    board_state: np.ndarray,
    current_player: int | None = None,
    weights_path: str | None = None,
    bot: Bot | None = None,
    **kwargs: Any,
) -> tuple[int, int]:
    if bot is not None:
        return bot.predict(board_state, current_player=current_player)
    b = Bot(weights_path=weights_path, **kwargs)
    return b.predict(board_state, current_player=current_player)
