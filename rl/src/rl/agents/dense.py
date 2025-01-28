from rl.agents import Agent, AgentConfig
from rl.memory.simple import SimpleMemory
from rust_reversi import Board
from rl.models.dense import DenseNet
import torch
import torchinfo
import random
import numpy as np

class DenseAgentConfig(AgentConfig):
    hidden_size: int


class DenseAgent(Agent):
    def __init__(self, config: DenseAgentConfig):
        super().__init__(config)
        self.memory = SimpleMemory(config["memory_size"])
        self.net = DenseNet(128, config["hidden_size"], 64)
        self.target_net = DenseNet(128, config["hidden_size"], 64)
        self.target_net.load_state_dict(self.net.state_dict())
        self.net.to(config["device"])
        self.target_net.to(config["device"])
        if config["verbose"]:
            torchinfo.summary(self.net, input_size=(config["batch_size"], 128), device=config["device"])
            for param in self.net.parameters():
                print(f"Device: {param.device}")
                break
        self.config = config
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config["lr"])
        self.criterion = torch.nn.SmoothL1Loss()

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

    def get_action(self, board: Board, episode: int) -> int:
        epsilon = self.get_epsilon(episode)
        if random.random() < epsilon:
            return random.choice(board.get_legal_moves_vec())
        self.net.eval()
        board_tensor = self.board_to_input(board)
        board_tensor = board_tensor.to(self.config["device"])
        with torch.no_grad():
            out: torch.Tensor = self.net(board_tensor)
        legal_actions: torch.Tensor = torch.tensor(board.get_legal_moves_tf(), dtype=torch.bool, device=self.config["device"])
        out = out.masked_fill(~legal_actions, -1e9)
        return out.argmax().item()
    
    def get_action_batch(self, boards: list[Board], episoide: int) -> list[int]:
        self.net.eval()
        board_tensors = torch.stack([self.board_to_input(board) for board in boards])
        board_tensors = board_tensors.to(self.config["device"])
        with torch.no_grad():
            out: torch.Tensor = self.net(board_tensors)
        legal_actions = torch.stack([torch.tensor(board.get_legal_moves_tf(), dtype=torch.bool, device=self.config["device"]) for board in boards])
        out = out.masked_fill(~legal_actions, -1e9)
        actions = out.argmax(dim=1).tolist()
        # override with epsilon greedy
        epsilon = self.get_epsilon(episoide)
        for i, board in enumerate(boards):
            if random.random() < epsilon:
                actions[i] = random.choice(board.get_legal_moves_vec())
        return actions
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
