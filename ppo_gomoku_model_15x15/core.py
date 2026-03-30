import torch
import torch.nn.functional as F


def compute_done(board, kernel_horizontal, kernel_vertical, kernel_diagonal):
    """
    Determine whether each environment has a winner.

    It checks horizontal, vertical and diagonal five-in-a-row patterns.

    Args:
        board: Board tensor of shape (E, B, B).
        kernel_horizontal: Horizontal convolution kernel.
        kernel_vertical: Vertical convolution kernel.
        kernel_diagonal: Diagonal convolution kernels.

    Returns:
        Boolean tensor of shape (E,).
    """

    board = board.unsqueeze(1)  # (E,1,B,B)

    output_horizontal = F.conv2d(
        input=board, weight=kernel_horizontal)  # (E,1,B-4,B)

    done_horizontal = (output_horizontal.flatten(
        start_dim=1) > 4.5).any(dim=-1)  # (E,)

    output_vertical = F.conv2d(
        input=board, weight=kernel_vertical)  # (E,1,B,B-4)

    done_vertical = (output_vertical.flatten(
        start_dim=1) > 4.5).any(dim=-1)  # (E,)

    output_diagonal = F.conv2d(
        input=board, weight=kernel_diagonal)  # (E,2,B-4,B-4)

    done_diagonal = (output_diagonal.flatten(
        start_dim=1) > 4.5).any(dim=-1)  # (E,)

    done = done_horizontal | done_vertical | done_diagonal

    return done


class Gomoku:
    def __init__(self, num_envs, board_size=15, device=None):
        """
        Initialise a batch of parallel Gomoku environments.

        Args:
            num_envs: Number of parallel environments.
            board_size: Side length of the square board.
            device: Torch device for tensors.
        """
        assert num_envs > 0
        assert board_size >= 5

        self.num_envs = num_envs
        self.board_size = board_size
        self.device = device
        # board 0 empty 1 black -1 white
        self.board = torch.zeros(
            num_envs,
            self.board_size,
            self.board_size,
            device=self.device,
            dtype=torch.long,
        )  # (E,B,B)
        self.done = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.turn = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self.move_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self.last_move = -torch.ones(num_envs, dtype=torch.long, device=self.device)

        self.kernel_horizontal = (
            torch.tensor([1, 1, 1, 1, 1], device=self.device,
                         dtype=torch.float)
            .unsqueeze(-1)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1,1,5,1)

        self.kernel_vertical = (
            torch.tensor([1, 1, 1, 1, 1], device=self.device,
                         dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1,1,1,5)

        self.kernel_diagonal = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ],
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ],
            ],
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(
            1
        )  # (2,1,5,5)

    def to(self, device):
        """
        Move internal tensors to the given device.

        Args:
            device: Target device.

        Returns:
            Self with tensors moved.
        """
        self.board.to(device=device)
        self.done.to(device=device)
        self.turn.to(device=device)
        self.move_count.to(device=device)
        self.last_move.to(device=device)
        return self

    def reset(self, env_indices=None):
        """
        Reset selected environments to the initial state.

        Args:
            env_indices: Optional environment indices to reset.
        """
        if env_indices is None:
            self.board.zero_()
            self.done.zero_()
            self.turn.zero_()
            self.move_count.zero_()
            self.last_move.fill_(-1)
        else:
            self.board[env_indices] = 0
            self.done[env_indices] = False
            self.turn[env_indices] = 0
            self.move_count[env_indices] = 0
            self.last_move[env_indices] = -1

    def step(self, action, env_mask=None):
        """
        Apply one action per environment and update game state.

        Args:
            action: Linear move positions with shape (E,).
            env_mask: Optional boolean mask of active environments.

        Returns:
            Tuple (done_statuses, invalid_actions).
        """

        if env_mask is None:
            env_mask = torch.ones_like(action, dtype=torch.bool)

        board_1d_view = self.board.view(self.num_envs, -1)

        values_on_board = board_1d_view[
            torch.arange(self.num_envs, device=self.device),
            action,
        ]  # (E,)

        nop = (values_on_board != 0) | (~env_mask)  # (E,)
        inc = torch.logical_not(nop).long()  # (E,)
        piece = torch.where(self.turn == 0, 1, -1)
        board_1d_view[
            torch.arange(self.num_envs, device=self.device), action
        ] = torch.where(nop, values_on_board, piece)
        self.move_count = self.move_count + inc

        # F.conv2d doesn't support LongTensor on CUDA. So we use float.
        board_one_side = (
            self.board == piece.unsqueeze(-1).unsqueeze(-1)).float()
        self.done = compute_done(
            board_one_side,
            self.kernel_horizontal,
            self.kernel_vertical,
            self.kernel_diagonal,
        ) | (self.move_count == self.board_size * self.board_size)

        self.turn = (self.turn + inc) % 2
        self.last_move = torch.where(nop, self.last_move, action)

        return self.done & env_mask, nop & env_mask

    def get_encoded_board(self):
        """
        Encode board state for neural-network input.

        Returns:
            Tensor of shape (E, 3, B, B).
        """
        piece = torch.where(self.turn == 0, 1, -
                            1).unsqueeze(-1).unsqueeze(-1)  # (E,1,1)

        layer1 = (self.board == piece).float()
        layer2 = (self.board == -piece).float()

        last_x = self.last_move // self.board_size  # (E,)
        last_y = self.last_move % self.board_size  # (E,)

        # (1,B)==(E,1)-> (E,B)-> (E,B,1)
        # (1,B)==(E,1)-> (E,B)-> (E,1,B)
        layer3 = (
            (
                torch.arange(self.board_size, device=self.device).unsqueeze(0)
                == last_x.unsqueeze(-1)
            ).unsqueeze(-1)
        ) & (
            (
                torch.arange(self.board_size, device=self.device).unsqueeze(0)
                == last_y.unsqueeze(-1)
            ).unsqueeze(1)
        )  # (E,B,B)
        layer3 = layer3.float()

        # layer4 = (self.turn == 0).float().unsqueeze(-1).unsqueeze(-1)  # (E,1,1)
        # layer4 = layer4.expand(-1, self.board_size, self.board_size)

        output = torch.stack(
            [
                layer1,
                layer2,
                layer3,
                # layer4,
            ],
            dim=1,
        )  # (E,*,B,B)
        return output

    def get_action_mask(self):
        """
        Return a mask of legal actions for each environment.

        Returns:
            Tensor of shape (E, B*B), True for legal actions.
        """
        return (self.board == 0).flatten(start_dim=1)

    def is_valid(self, action):
        """
        Check whether each action is valid.

        Args:
            action: Linear indexed actions.

        Returns:
            Boolean tensor of shape (E,).
        """
        out_of_range = action < 0 | (
            action >= self.board_size * self.board_size)
        x = action // self.board_size
        y = action % self.board_size

        values_on_board = self.board[
            torch.arange(self.num_envs, device=self.device), x, y
        ]  # (E,)

        not_empty = values_on_board != 0  # (E,)

        invalid = out_of_range | not_empty

        return ~invalid
