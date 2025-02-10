import numpy as np
import torch
from rust_reversi import Board
from distillation.models import ReversiNet

INPUT_SIZE = 128
# 8x8 board + 1 for pass
OUTPUT_SIZE = 65
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DenseNet(ReversiNet):
    def __init__(self, hidden_size: int):
        super(DenseNet, self).__init__()
        self.fc1 = torch.nn.Linear(INPUT_SIZE, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, OUTPUT_SIZE)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x

    def get_action(self, board: Board) -> int:
        board_tensor = torch.zeros(128, dtype=torch.float32)
        player_board, opponent_board, _turn = board.get_board()
        for i in range(64):
            bit = 1 << (64 - i - 1)
            if player_board & bit:
                board_tensor[i] = 1.0
            if opponent_board & bit:
                board_tensor[i + 64] = 1.0
        board_tensor = board_tensor.to(DEVICE)
        with torch.no_grad():
            output = self.forward(board_tensor)
        legal_actions = torch.tensor(
            board.get_legal_moves_tf() + [board.is_pass()],
            dtype=torch.bool,
            device=DEVICE,
        )
        output.masked_fill_(~legal_actions, -1e9)
        return torch.argmax(output).item()

    @staticmethod
    def x2input(x: np.void) -> torch.Tensor:
        res = torch.zeros(128, dtype=torch.float32)
        player_board = x["player_board"]
        opponent_board = x["opponent_board"]
        for i in range(64):
            bit = 1 << (64 - i - 1)
            if player_board & bit:
                res[i] = 1.0
            if opponent_board & bit:
                res[i + 64] = 1.0
        return res
