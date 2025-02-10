import numpy as np
import torch
from abc import ABC, abstractmethod

class ReversiNet(torch.nn.Module, ABC):
    @abstractmethod
    def get_action(self, board) -> int:
        pass
    @abstractmethod
    def x2input(x: np.void) -> torch.Tensor:
        pass
