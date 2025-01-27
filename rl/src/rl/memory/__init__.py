from abc import ABC, abstractmethod
from typing import List, Tuple
from rust_reversi import Board

class Memory(ABC):
    @abstractmethod
    def push(self, state: Board, action: int, next_state: Board, reward: float) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[Tuple[Board, int, Board, float]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
