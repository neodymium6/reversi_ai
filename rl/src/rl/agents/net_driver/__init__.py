from abc import ABC, abstractmethod
import torch
from rust_reversi import Board
from typing import List, TypedDict
from enum import Enum

import torchinfo

class NetType(Enum):
    RESNET10 = 0
    Conv5Dueling = 1
    Conv5 = 2
    Dense = 3

class NetConfig(TypedDict):
    net_type: NetType

class NetDriver(ABC):
    def __init__(self, verbose: bool, device: torch.device, config: NetConfig, batch_size: int):
        self.verbose = verbose
        self.device = device
        self.config = config
        self.batch_size = batch_size
        self.net: torch.nn.Module = None
        self.target_net: torch.nn.Module = None

    def after_init(self):
        self.net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        if self.verbose:
            for param in self.net.parameters():
                print(f"Device: {param.device}")
                break

    @abstractmethod
    def board_to_input(self, board: Board) -> torch.Tensor:
        pass

    @abstractmethod
    def get_action(self, board: Board, episilon: float) -> int:
        pass

    @abstractmethod
    def get_action_batch(self, boards: List[Board], epsilon: float) -> List[int]:
        pass

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def save(self, path: str):
        if self.verbose:
            print(f"Saving model to {path}")
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        if self.verbose:
            print(f"Loading model from {path}")
        self.net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(torch.load(path, weights_only=True))
