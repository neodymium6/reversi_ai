from rl.agents import AgentConfig
from rl.agents.q_net import QnetAgent
from rl.memory.simple import SimpleMemory
from rust_reversi import Board
from rl.models.dense import DenseNet
import torch
import torchinfo

class DenseAgentConfig(AgentConfig):
    hidden_size: int


class DenseAgent(QnetAgent):
    def __init__(self, config: DenseAgentConfig):
        super().__init__(config)
        self.memory = SimpleMemory(config["memory_size"])
        self.net = DenseNet(config["hidden_size"])
        self.target_net = DenseNet(config["hidden_size"])
        if config["verbose"]:
            torchinfo.summary(self.net, input_size=(config["batch_size"], 128), device=config["device"])
        self.config = config
        super().after_init()

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
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
