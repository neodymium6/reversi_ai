from abc import ABC, abstractmethod
from rust_reversi import Board

class Agent(ABC):
    @abstractmethod
    def get_action(self, board: Board, episoide: int) -> int:
        pass
