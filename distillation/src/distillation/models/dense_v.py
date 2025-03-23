import base64
import struct
import numpy as np
import torch
from rust_reversi import Board
from distillation.models import ReversiNet

INPUT_SIZE = 128
# 8x8 board + 1 for pass
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DenseNetV(ReversiNet):
    def __init__(self, hidden_size: int):
        super(DenseNetV, self).__init__()
        self.fc1 = torch.nn.Linear(INPUT_SIZE, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x

    def board2input(self, board: Board) -> torch.Tensor:
        board_tensor = torch.zeros(128, dtype=torch.float32)
        player_board, opponent_board, _turn = board.get_board()
        for i in range(64):
            bit = 1 << (64 - i - 1)
            if player_board & bit:
                board_tensor[i] = 1.0
            if opponent_board & bit:
                board_tensor[i + 64] = 1.0
        return board_tensor

    def get_action(self, board: Board) -> int:
        legal_actions = board.get_legal_moves_vec()
        next_boards = []
        for action in legal_actions:
            tmp_board = board.clone()
            tmp_board.do_move(action)
            next_boards.append(tmp_board)
        board_tensors = torch.stack([self.board2input(tmp_board) for tmp_board in next_boards])
        board_tensors = board_tensors.to(DEVICE)
        with torch.no_grad():
            values = self.forward(board_tensors)
        values = values.view(-1)
        values = -values
        _best_value, best_index = torch.max(values, 0)
        best_action = legal_actions[best_index.item()]
        return best_action

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

    def save_weights_base64(self) -> str:
        # 重みを取得
        ih_weights = self.fc1.weight.data.cpu().numpy().astype(np.float32)
        h_biases = self.fc1.bias.data.cpu().numpy().astype(np.float32)
        ho_weights = self.fc2.weight.data.cpu().numpy().squeeze().astype(np.float32)
        o_bias = self.fc2.bias.data.cpu().numpy().astype(np.float32)[0]
        
        # バイナリに変換
        binary = b''
        binary += ih_weights.tobytes()
        binary += h_biases.tobytes()
        binary += ho_weights.tobytes()
        binary += struct.pack('f', o_bias)
        
        # base64エンコード
        return base64.b64encode(binary).decode('ascii')
