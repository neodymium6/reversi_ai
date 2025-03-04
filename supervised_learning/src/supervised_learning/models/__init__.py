from abc import ABC, abstractmethod
import torch
from rust_reversi import Board

class ReversiNet(torch.nn.Module, ABC):
    @abstractmethod
    def get_action(self, board: Board) -> int:
        pass
    @abstractmethod
    def board_to_input(self, board: Board) -> torch.Tensor:
        pass
    @abstractmethod
    def save_weights_base64(self) -> str:
        pass
