from abc import abstractmethod
import random
from typing import List
from rust_reversi import Board
import torch
from rl.agents.net_driver import NetDriver, NetConfig

class QnetDriver(NetDriver):
    def __init__(self, verbose: bool, device: torch.device, config: NetConfig, batch_size: int):
        super().__init__(verbose, device, config, batch_size)

    @abstractmethod
    def board_to_input(self, board: Board) -> torch.Tensor:
        pass

    def get_action(self, board: Board, epsilon: float) -> int:
        if board.is_pass():
            return 64
        if random.random() < epsilon:
            return random.choice(board.get_legal_moves_vec())
        self.net.eval()
        board_tensor = self.board_to_input(board)
        board_tensor = board_tensor.to(self.device)
        with torch.no_grad():
            out: torch.Tensor = self.net(board_tensor)
        # 64th element is pass
        legal_actions: torch.Tensor = torch.tensor(
            board.get_legal_moves_tf() + [board.is_pass()],
            dtype=torch.bool,
            device=self.device,
        )
        out = out.masked_fill(~legal_actions, -1e9)
        return out.argmax().item()

    def get_action_batch(self, boards: List[Board], epsilon: float) -> List[int]:
        self.net.eval()
        board_tensors = torch.stack([self.board_to_input(board) for board in boards])
        board_tensors = board_tensors.to(self.device)
        with torch.no_grad():
            out: torch.Tensor = self.net(board_tensors)
        # 64th element is pass
        legal_actions = torch.stack([torch.tensor(
                board.get_legal_moves_tf() + [board.is_pass()],
                dtype=torch.bool,
                device=self.device
        ) for board in boards])
        out = out.masked_fill(~legal_actions, -1e9)
        actions = out.argmax(dim=1).tolist()
        # override with epsilon greedy
        for i, board in enumerate(boards):
            if board.is_pass():
                actions[i] = 64
                continue
            if random.random() < epsilon:
                actions[i] = random.choice(board.get_legal_moves_vec())
        return actions
