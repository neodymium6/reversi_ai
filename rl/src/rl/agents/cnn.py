from rl.agents import AgentConfig
from rl.agents.q_net import QnetAgent
from rl.memory.simple import SimpleMemory
from rust_reversi import Board
from rl.models.cnn_dueling import Conv5DuelingNet
import torch
import torchinfo

class CnnAgentConfig(AgentConfig):
    num_channels: int
    fc_hidden_size: int


class CnnAgent(QnetAgent):
    def __init__(self, config: CnnAgentConfig):
        super().__init__(config)
        self.memory = SimpleMemory(config["memory_size"])
        self.net = Conv5DuelingNet(config["num_channels"], config["fc_hidden_size"])
        self.target_net = Conv5DuelingNet(config["num_channels"], config["fc_hidden_size"])
        if config["verbose"]:
            torchinfo.summary(self.net, input_size=(config["batch_size"], 2, 8, 8), device=config["device"])
        self.config = config
        super().after_init()

    def board_to_input(self, board: Board) -> torch.Tensor:
        board_matrix = board.get_board_matrix()
        board_tensor = torch.tensor(board_matrix, dtype=torch.float32)
        board_tensor = board_tensor[0:2, :, :]
        return board_tensor
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
