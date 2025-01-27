from rl.agents import Agent
from rl.memory.simple import SimpleMemory
from rust_reversi import Board
from rl.models.dense import DenseNet
from typing import List, TypedDict
import torch
import torchinfo

class DenseAgentConfig(TypedDict):
    memory_size: int
    hidden_size: int
    device: torch.device
    verbose: bool
    batch_size: int


class DenseAgent(Agent):
    def __init__(self, config: DenseAgentConfig):
        super().__init__()
        self.memory = SimpleMemory(config["memory_size"])
        self.net = DenseNet(128, config["hidden_size"], 64)
        self.net.to(config["device"])
        if config["verbose"]:
            torchinfo.summary(self.net, input_size=(config["batch_size"], 128), device=config["device"])
            for param in self.net.parameters():
                print(f"Device: {param.device}")
                break
        self.config = config

    def board_to_input(self, board: Board) -> torch.Tensor:
        res = torch.zeros(128, dtype=torch.float32)
        player_board, opponent_board, _turn = board.get_board()
        for i in range(64):
            bit = 1 << (64 - i - 1)
            if player_board & bit:
                res[i] = 1.0
            if opponent_board & bit:
                res[i + 64] = 1.0
        return res

    def get_action(self, board: Board) -> int:
        board_tensor = self.board_to_input(board)
        board_tensor = board_tensor.to(self.config["device"])
        with torch.no_grad():
            out: torch.Tensor = self.net(board_tensor)
        legal_actions: List[bool] = board.get_legal_moves_tf()
        legal_actions = [1 if x else 0 for x in legal_actions]
        out = out.cpu().numpy()
        out = out * legal_actions
        return out.argmax()
