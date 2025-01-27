from abc import ABC, abstractmethod
from rust_reversi import Board

class Memory(ABC):
    @abstractmethod
    def push(self, state: Board, action: int, next_state: Board, reward: float) -> None:
        pass