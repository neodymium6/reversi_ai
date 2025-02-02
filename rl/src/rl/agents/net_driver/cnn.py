from rust_reversi import Board
import torch
import torchinfo
from rl.agents.net_driver.q_net import QnetDriver
from rl.agents.net_driver import NetConfig, NetType
from rl.models.resnet import ResNet10
from rl.models.cnn_dueling import Conv5DuelingNet
from rl.models.cnn import Conv5Net
from rl.models.transformer import Transformer

class CnnConfig(NetConfig):
    num_channels: int
    fc_hidden_size: int

class CnnDriver(QnetDriver):
    def __init__(self, verbose: bool, device: torch.device, config: CnnConfig, batch_size: int):
        super().__init__(verbose, device, config, batch_size)
        if config["net_type"] == NetType.RESNET10:
            self.net = ResNet10(config["num_channels"], config["fc_hidden_size"])
            self.target_net = ResNet10(config["num_channels"], config["fc_hidden_size"])
        elif config["net_type"] == NetType.Conv5Dueling:
            self.net = Conv5DuelingNet(config["num_channels"], config["fc_hidden_size"])
            self.target_net = Conv5DuelingNet(config["num_channels"], config["fc_hidden_size"])
        elif config["net_type"] == NetType.Conv5:
            self.net = Conv5Net(config["num_channels"], config["fc_hidden_size"])
            self.target_net = Conv5Net(config["num_channels"], config["fc_hidden_size"])
        elif config["net_type"] == NetType.Transformer:
            self.net = Transformer()
            self.target_net = Transformer()
        else:
            raise ValueError(f"Invalid net type: {config['net_type']}")
        if verbose:
            torchinfo.summary(self.net, input_size=(batch_size, 2, 8, 8), device=device)
        self.after_init()

    def board_to_input(self, board: Board) -> torch.Tensor:
        board_matrix = board.get_board_matrix()
        board_tensor = torch.tensor(board_matrix, dtype=torch.float32)
        board_tensor = board_tensor[0:2, :, :]
        return board_tensor
