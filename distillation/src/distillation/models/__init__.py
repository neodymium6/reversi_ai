import torch
from abc import ABC, abstractmethod

class StudentNet(torch.nn.Module, ABC):
    @abstractmethod
    def get_action(self, board) -> int:
        pass
