from rust_reversi import Board
import torch
import torchinfo
from rl.agents.net_driver.q_net import QnetDriver
from rl.models.dense import DenseNet
from rl.agents.net_driver import NetConfig

class DenseConfig(NetConfig):
    hidden_size: int

class DenseDriver(QnetDriver):
    def __init__(self, verbose: bool, device: torch.device, config: DenseConfig, batch_size: int):
        super().__init__(verbose, device, config, batch_size)
        self.net = DenseNet(config["hidden_size"])
        self.target_net = DenseNet(config["hidden_size"])
        if verbose:
            torchinfo.summary(self.net, input_size=(batch_size, 128), device=device)
        self.after_init()

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
