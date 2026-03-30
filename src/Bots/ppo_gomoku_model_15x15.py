from __future__ import annotations

import os
import random

import numpy as np

from src.Bots.base_bot import BaseBot
from src.game_datatypes import GameState

try:
    import torch
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from tensordict.nn import InteractionType, set_interaction_type

    from ppo_gomoku_model_15x15.env import GomokuEnv
    from ppo_gomoku_model_15x15.policy import get_policy

    _PPO_BACKEND_AVAILABLE = True
    _PPO_BACKEND_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    torch = None
    OmegaConf = None
    TensorDict = None
    InteractionType = None
    set_interaction_type = None
    GomokuEnv = None
    get_policy = None
    _PPO_BACKEND_AVAILABLE = False
    _PPO_BACKEND_IMPORT_ERROR = exc


class Bot(BaseBot):
    """
    RL bot wrapper around pre-trained PPO policies.

    If PPO dependencies or files are unavailable, this class falls back
    to the heuristic tactical bot by default so main UI does not crash.
    Pass strict=True to force hard failures instead.
    """

    _printed_fallback_warning = False

    def __init__(
        self,
        board_size=15,
        algo="ppo",
        weights_path=None,
        device=None,
        deterministic=True,
        strict: bool = False,
    ):
        super().__init__()

        self.board_size = int(board_size)
        self.algo = str(algo).lower()
        self.deterministic = bool(deterministic)
        self.policy = None
        self._fallback_bot = None
        self._fallback_reason = None

        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
        )

        if self.algo != "ppo":
            reason = f"Unsupported algo='{self.algo}' in PPO wrapper."
            if strict:
                raise ValueError(reason)
            self._enable_fallback(reason)
            return

        if not _PPO_BACKEND_AVAILABLE:
            reason = (
                "PPO optional dependencies unavailable "
                f"({_PPO_BACKEND_IMPORT_ERROR})."
            )
            if strict:
                raise RuntimeError(reason) from _PPO_BACKEND_IMPORT_ERROR
            self._enable_fallback(reason)
            return

        if device is None:
            chosen = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            chosen = str(device)
            if chosen.startswith("cuda") and not torch.cuda.is_available():
                chosen = "cpu"
        self.device = torch.device(chosen)

        if weights_path is None:
            weights_path = os.path.join(
                self.project_root,
                "pretrained_models",
                f"{self.board_size}_{self.board_size}",
                self.algo,
                "0.pt",
            )

        if not os.path.isabs(weights_path):
            weights_path = os.path.join(self.project_root, weights_path)
        self.weights_path = weights_path

        try:
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {self.weights_path}. "
                    "Pass --weights_path explicitly or use board_size=15 with bundled checkpoints."
                )

            cfg_path = os.path.join(self.project_root, "cfg", "algo", f"{self.algo}.yaml")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"Algorithm config not found: {cfg_path}")
            cfg = OmegaConf.load(cfg_path)

            # PPO setup here expects batch size > 1 on initialisation.
            # We create specs with two envs and duplicate inputs at inference time.
            spec_env = GomokuEnv(num_envs=2, board_size=self.board_size, device=str(self.device))
            self.policy = get_policy(
                name=self.algo,
                cfg=cfg,
                action_spec=spec_env.action_spec,
                observation_spec=spec_env.observation_spec,
                device=str(self.device),
            )
            self.policy.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            self.policy.eval()
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            self._enable_fallback(f"PPO init failed: {exc}")
            return

    def _enable_fallback(self, reason: str) -> None:
        self._fallback_reason = reason
        fallback_name = "heuristic_tactical.predict"
        try:
            from src.Bots.heuristic_tactical import predict as heuristic_predict
            from src.gomoku_game import get_legal_moves

            class _PredictFallbackBot:
                def move(self, game_state: GameState):
                    current_player = game_state.current_player
                    if current_player is None:
                        p0 = int(np.sum(game_state.board == 0))
                        p1 = int(np.sum(game_state.board == 1))
                        current_player = 0 if p0 <= p1 else 1
                    try:
                        return heuristic_predict(game_state.board.copy(), int(current_player))
                    except Exception:  # noqa: BLE001
                        legal = get_legal_moves(game_state.board)
                        return random.choice(legal) if legal else None

            self._fallback_bot = _PredictFallbackBot()
        except Exception:  # noqa: BLE001
            from src.Bots.random import Bot as RandomBot

            self._fallback_bot = RandomBot()
            fallback_name = "random.Bot"
        if not Bot._printed_fallback_warning:
            print(
                f"[ppo_gomoku_model_15x15] fallback -> {fallback_name} "
                f"(reason: {reason})"
            )
            Bot._printed_fallback_warning = True

    def _encode(self, game_state):
        board = game_state.board

        current_player = game_state.current_player
        if current_player is None:
            # Infer turn from stone counts if not already set.
            count_0 = int(np.sum(board == 0))
            count_1 = int(np.sum(board == 1))
            current_player = 0 if count_0 <= count_1 else 1

        current = (board == current_player).astype(np.float32)
        opponent = ((board != -1) & (board != current_player)).astype(np.float32)

        last = np.zeros_like(current, dtype=np.float32)
        last_move = getattr(game_state, "last_move", None)
        if isinstance(last_move, tuple) and len(last_move) == 2:
            x, y = int(last_move[0]), int(last_move[1])
            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                last[x, y] = 1.0

        obs = np.stack([current, opponent, last], axis=0)
        action_mask = (board == -1).reshape(-1).astype(np.bool_)
        return obs, action_mask

    def _fallback_legal(self, board):
        legal = np.argwhere(board == -1)
        if legal.size == 0:
            return None
        idx = random.randint(0, len(legal) - 1)
        x, y = legal[idx]
        return int(x), int(y)

    def move(self, game_state):
        if self._fallback_bot is not None:
            return self._fallback_bot.move(game_state)

        obs, action_mask = self._encode(game_state)

        obs_t = torch.from_numpy(obs).to(self.device)
        mask_t = torch.from_numpy(action_mask).to(self.device)
        td = TensorDict(
            {
                "observation": torch.stack([obs_t, obs_t], dim=0),
                "action_mask": torch.stack([mask_t, mask_t], dim=0),
            },
            batch_size=[2],
            device=self.device,
        )

        interaction = InteractionType.MODE if self.deterministic else InteractionType.RANDOM

        with torch.no_grad(), set_interaction_type(interaction):
            self.policy.eval()
            td = self.policy(td)

        action = int(td["action"][0].item())
        x, y = divmod(action, self.board_size)
        if game_state.board[x, y] != -1:
            return self._fallback_legal(game_state.board)
        return x, y
