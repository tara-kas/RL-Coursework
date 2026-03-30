from tensordict import TensorDict
import torch

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from .core import Gomoku

class GomokuEnv:
    def __init__(self, num_envs, board_size, device=None):
        """
        Initialise a parallel Gomoku environment.

        Args:
            num_envs: Number of parallel game environments.
            board_size: Size of the square board.
            device: Torch device, for example CPU or GPU.
        """
        self.gomoku = Gomoku(num_envs=num_envs, board_size=board_size, device=device)

        self.observation_spec = CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    device=self.device,
                    shape=[num_envs, 3, board_size, board_size],
                ),
                "action_mask": BinaryDiscreteTensorSpec(
                    n=board_size * board_size,
                    device=self.device,
                    shape=[num_envs, board_size * board_size],
                    dtype=torch.bool,
                ),
            },
            shape=[
                num_envs,
            ],
            device=self.device,
        )
        self.action_spec = DiscreteTensorSpec(
            board_size * board_size,
            shape=[
                num_envs,
            ],
            device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=[num_envs, 1],
            device=self.device,
        )

        self._post_step = None

    @property
    def batch_size(self):
        return torch.Size((self.num_envs,))

    @property
    def board_size(self):
        return self.gomoku.board_size

    @property
    def device(self):
        return self.gomoku.device

    @property
    def num_envs(self):
        return self.gomoku.num_envs

    def reset(self, env_indices=None):
        """
        Reset selected environments, or all environments when no indices are passed.

        Args:
            env_indices: Environment indices to reset.

        Returns:
            TensorDict with observations and action masks.
        """
        self.gomoku.reset(env_indices=env_indices)
        tensordict = TensorDict(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
            },
            self.batch_size,
            device=self.device,
        )
        return tensordict

    def step(self, tensordict):
        """
        Step all selected environments by one move.

        Args:
            tensordict: TensorDict containing actions and optional env mask.

        Returns:
            TensorDict with updated observations, masks and step stats.
        """
        action = tensordict.get("action")
        env_mask = tensordict.get("env_mask", None)
        episode_len = self.gomoku.move_count + 1  # (E,)
        win, illegal = self.gomoku.step(action=action, env_mask=env_mask)

        assert not illegal.any()

        done = win
        black_win = win & (episode_len % 2 == 1)
        white_win = win & (episode_len % 2 == 0)
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(
            {
                "observation": self.gomoku.get_encoded_board(),
                "action_mask": self.gomoku.get_action_mask(),
                "done": done,
                "win": win,
                # reward is calculated later
                "stats": {
                    "episode_len": episode_len,
                    "black_win": black_win,
                    "white_win": white_win,
                },
            }
        )
        if self._post_step:
            self._post_step(tensordict)
        return tensordict

    def step_and_maybe_reset(self, tensordict, env_mask=None):
        """
        Step once and reset finished environments.

        Args:
            tensordict: TensorDict with current state and actions.
            env_mask: Optional mask of environments to step.

        Returns:
            TensorDict with updated state.
            Finished environments are reset but still marked done for this step.
        """

        if env_mask is not None:
            tensordict.set("env_mask", env_mask)
        next_tensordict = self.step(tensordict=tensordict)
        tensordict.exclude("env_mask", inplace=True)

        done = next_tensordict.get("done")  # (E,)
        env_ids = done.nonzero().squeeze(0)
        reset_td = self.reset(env_indices=env_ids)
        next_tensordict.update(reset_td)  # no impact on training
        return next_tensordict

    def set_post_step(self, post_step=None):
        """
        Set a callback to run after each environment step.

        Args:
            post_step: Callable that receives the step TensorDict.
        """
        self._post_step = post_step
