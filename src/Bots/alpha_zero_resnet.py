import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Bots.base_bot import BaseBot
from src.game_datatypes import GameState
from src.gomoku_game import get_legal_moves
from src.mcts import MCTSNode, run_mcts, run_mcts_with_policy
from src.model_loader import DEFAULT_WEIGHTS_PATH, load_weights, save_weights as save_weights_to_path


def _maybe_compile(model: nn.Module, enable: bool = True) -> nn.Module:
    """Compile model with torch.compile if PyTorch >= 2 and enable is True. First run may be slow (tracing)."""
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
    """Standard AlphaZero Residual Block."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroTransform(nn.Module):
    """
    Renamed internally to use a ResNet architecture (AlphaZero standard)
    while keeping the class name identical to avoid breaking imports in train.py.
    """
    def __init__(self, board_size=15, num_channels=128, num_res_blocks=6):
        super().__init__()
        self.board_size = board_size
        
        # Initial Convolutional Block
        self.conv_input = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual Blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, 3, H, W)
        mask: optional (batch, 225) where 1=legal, 0=illegal. If None, no masking.
        """
        # Common Representation
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head (Outputs probability distribution over moves)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) # Flatten spatial dimensions
        logits = self.policy_fc(p)
        
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e4)
            
        policy = F.softmax(logits, dim=-1)
        
        # Value Head (Outputs scalar -1 to 1)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1) # Flatten to preserve location data!
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value


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
            self.model = AlphaZeroTransform(board_size=board_size).to(self.device)
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
        move = run_mcts(
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
        return move

    def get_move_and_policy(
        self,
        board: np.ndarray,
        current_player: int,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        add_root_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        root: MCTSNode | None = None,
        return_root: bool = False,
    ) -> tuple[tuple[int, int], np.ndarray] | tuple[tuple[int, int], np.ndarray, MCTSNode | None]:
        move, policy, next_root = run_mcts_with_policy(
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
            root=root,
        )
        if return_root:
            return move, policy, next_root
        return move, policy

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
