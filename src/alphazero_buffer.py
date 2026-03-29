from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SelfPlayData:
    boards: np.ndarray
    players: np.ndarray
    policies: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return int(self.boards.shape[0])

    @classmethod
    def empty(cls, board_size: int) -> "SelfPlayData":
        policy_size = board_size * board_size
        return cls(
            boards=np.empty((0, board_size, board_size), dtype=np.int8),
            players=np.empty((0,), dtype=np.uint8),
            policies=np.empty((0, policy_size), dtype=np.float16),
            values=np.empty((0,), dtype=np.float16),
        )


class AlphaZeroReplayBuffer:
    def __init__(self, capacity: int, board_size: int) -> None:
        self.capacity = capacity
        self.board_size = board_size
        policy_size = board_size * board_size
        self.boards = np.empty((capacity, board_size, board_size), dtype=np.int8)
        self.players = np.empty((capacity,), dtype=np.uint8)
        self.policies = np.empty((capacity, policy_size), dtype=np.float16)
        self.values = np.empty((capacity,), dtype=np.float16)
        self.size = 0
        self.write_index = 0

    def __len__(self) -> int:
        return self.size

    def extend(self, data: SelfPlayData) -> None:
        n_items = len(data)
        if n_items == 0:
            return

        if n_items >= self.capacity:
            self.boards[:] = data.boards[-self.capacity :]
            self.players[:] = data.players[-self.capacity :]
            self.policies[:] = data.policies[-self.capacity :]
            self.values[:] = data.values[-self.capacity :]
            self.size = self.capacity
            self.write_index = 0
            return

        first_chunk = min(n_items, self.capacity - self.write_index)
        second_chunk = n_items - first_chunk

        end = self.write_index + first_chunk
        self.boards[self.write_index : end] = data.boards[:first_chunk]
        self.players[self.write_index : end] = data.players[:first_chunk]
        self.policies[self.write_index : end] = data.policies[:first_chunk]
        self.values[self.write_index : end] = data.values[:first_chunk]

        if second_chunk > 0:
            self.boards[:second_chunk] = data.boards[first_chunk:]
            self.players[:second_chunk] = data.players[first_chunk:]
            self.policies[:second_chunk] = data.policies[first_chunk:]
            self.values[:second_chunk] = data.values[first_chunk:]

        self.write_index = (self.write_index + n_items) % self.capacity
        self.size = min(self.capacity, self.size + n_items)

    def shuffled_indices(self) -> np.ndarray:
        return np.random.permutation(self.size)

    def get_batch(
        self,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.boards[indices],
            self.players[indices],
            self.policies[indices],
            self.values[indices],
        )
