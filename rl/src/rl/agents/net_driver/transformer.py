from rust_reversi import Board
import torch
import torchinfo
from rl.agents.net_driver.q_net import QnetDriver
from rl.agents.net_driver import NetConfig, NetType
from rl.models.transformer import Transformer

class TransformerConfig(NetConfig):
    patch_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_ratio: float
    dropout: float

class TransformerDriver(QnetDriver):
    def __init__(self, verbose: bool, device: torch.device, config: TransformerConfig, batch_size: int):
        super().__init__(verbose, device, config, batch_size)
        if config["net_type"] == NetType.Transformer:
            self.net = Transformer(config["patch_size"], config["embed_dim"], config["num_heads"], config["num_layers"], config["mlp_ratio"], config["dropout"])
            self.target_net = Transformer(config["patch_size"], config["embed_dim"], config["num_heads"], config["num_layers"], config["mlp_ratio"], config["dropout"])
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
