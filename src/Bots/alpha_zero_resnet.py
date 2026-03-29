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

import mcts_cpp

def run_fast_mcts_with_policy(
    tree,                  # <--- NEW: Accept the tree object
    board: np.ndarray,
    current_player: int,
    model: torch.nn.Module,
    last_move: tuple[int, int] | None = None, # <--- NEW: Accept the last move
    num_simulations: int = 200,
    batch_size: int = 64,
    c_puct: float = 1.5,
    device: torch.device = torch.device("cuda")
) -> tuple[tuple[int, int], np.ndarray, float]:
    
    if last_move is None:
        # Very first turn of the game
        tree.setRootState(board.astype(np.float32), current_player)
    else:
        # Convert 2D move to 1D move for C++
        move_1d = last_move[0] * 15 + last_move[1]
        tree.changeRoot(move_1d, board.astype(np.float32), current_player)

    sims_completed = 0
    while sims_completed < num_simulations:
        
        # --- PING (Gather from C++) ---
        batch_boards = tree.gather_batch(batch_size, c_puct)
        actual_batch = batch_boards.shape[0]
        
        if actual_batch == 0:
            # All paths hit game-over states!
            sims_completed += batch_size
            continue

        # --- PREPROCESS FOR RESNET ---
        # Convert raw (B, 15, 15) array to PyTorch tensor
        batch_tensor = torch.from_numpy(batch_boards).to(device)
        
        # Build the 3 planes: [Current Player Pieces, Opponent Pieces, Color to Play]
        # 1. Pre-allocate the memory exactly ONCE
        planes = torch.empty((actual_batch, 3, 15, 15), dtype=torch.float32, device=device)
        
        # 2. Fill the memory in-place (drastically reduces GPU overhead)
        planes[:, 0, :, :] = (batch_tensor == current_player).float()
        planes[:, 1, :, :] = (batch_tensor == (1 - current_player)).float()
        planes[:, 2, :, :] = float(current_player)
        
        # 3. Create the mask directly
        mask = (batch_tensor.view(actual_batch, 225) == -1).float()
        
        # Build the legal move mask (batch_size, 225) -> 1 is legal, 0 is illegal
        # -1 represents an empty square in your game logic
        mask = (batch_tensor.view(actual_batch, 225) == -1).float()

        # --- THE NET (Predict on GPU) ---
        with torch.inference_mode():
            policy, value = model(planes, mask)
            
            # Move back to CPU for C++
            policy_np = policy.cpu().numpy()
            value_np = value.view(-1).cpu().numpy()

        # --- PONG (Send back to C++) ---
        tree.expand_backup(policy_np, value_np)
        
        sims_completed += actual_batch

    # --- EXTRACT RESULTS ---
    # Get the visit counts from C++
    visits_1d = np.array(tree.get_root_visits(), dtype=np.float32)
    
    # Calculate Policy Target (pi)
    total_visits = visits_1d.sum()
    if total_visits == 0: 
        total_visits = 1
    policy_target = visits_1d / total_visits

    # Pick the best move (highest visit count)
    best_move_idx = int(np.argmax(visits_1d))
    best_move_2d = (best_move_idx // 15, best_move_idx % 15)

    root_value = tree.get_root_value()

    return best_move_2d, policy_target, root_value


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

        import mcts_cpp
        temp_tree = mcts_cpp.MCTS()

        best_move, _ = run_fast_mcts_with_policy(
            temp_tree,
            board_state.copy(),
            current_player,
            self.model,
            num_simulations=self.num_simulations,
            batch_size=self.mcts_batch_size,
            c_puct=1.5,
            device=self.device
        )
        
        return best_move

    def get_move_and_policy(
        self,
        board: np.ndarray,
        current_player: int,
        tree=None,
        last_move=None,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        add_root_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        override_simulations: int | None = None,
    ) -> tuple[tuple[int, int], np.ndarray]:
        sims_to_run = override_simulations if override_simulations is not None else self.num_simulations
        best_move, policy, root_value = run_fast_mcts_with_policy(
            tree=tree,
            board=board.copy(),
            current_player=current_player,
            model=self.model,
            last_move=last_move,
            num_simulations=sims_to_run,
            batch_size=self.mcts_batch_size,
            c_puct=c_puct,
            device=self.device
        )
        
        return best_move, policy, root_value

    def move(self, game_state: GameState) -> tuple[int, int] | None:
        legal_moves = get_legal_moves(game_state.board)
        if not legal_moves:
            return None

        current_player = game_state.current_player if game_state.current_player is not None else 0

        import mcts_cpp
        temp_tree = mcts_cpp.MCTS()
        
        best_move, _, _ = run_fast_mcts_with_policy(
            tree=temp_tree,
            board=game_state.board.copy(),
            current_player=current_player,
            model=self.model,
            num_simulations=self.num_simulations,
            batch_size=self.mcts_batch_size,
            c_puct=1.5,
            device=self.device
        )

        
        return best_move


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