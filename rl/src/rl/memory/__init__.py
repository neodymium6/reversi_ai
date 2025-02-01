from abc import ABC, abstractmethod
from typing import List, Tuple, TypedDict
from rust_reversi import Board
from enum import Enum

class MemoryType(Enum):
    PROPORTIONAL = 0
    UNIFORM = 1

class MemoryConfig(TypedDict):
    memory_size: int
    memory_type: MemoryType
    alpha: float
    beta: float

class Memory(ABC):
    @abstractmethod
    def push(self, state: Board, action: int, next_state: Board, reward: float) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Tuple[List[Tuple[Board, int, Board, float]], List[int], List[float]]:
        """
        Returns a tuple of a list of samples, a list of indices, and a list of weights.
        The list of samples is a list of tuples of (state, action, next_state, reward).
        The list of indices is a list of indices that were sampled.
        The list of weights is a list of weights for the samples
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
