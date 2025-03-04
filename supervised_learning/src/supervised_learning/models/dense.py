import base64
import struct
import numpy as np
import torch
from rust_reversi import Board
from supervised_learning.models import ReversiNet

INPUT_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DenseNet(ReversiNet):
    def __init__(self, hidden_size: int):
        super(DenseNet, self).__init__()
        self.fc1 = torch.nn.Linear(INPUT_SIZE, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x

    def board_to_input(self, board: Board) -> torch.Tensor:
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
        board_tensors = torch.stack([self.board_to_input(tmp_board) for tmp_board in next_boards])
        board_tensors = board_tensors.to(DEVICE)
        with torch.no_grad():
            values = self.forward(board_tensors)
        values = values.view(-1)
        values = 1.0 - values
        values = torch.clamp(values, 0.0, 1.0)
        _best_value, best_index = torch.max(values, 0)
        best_action = legal_actions[best_index.item()]
        return best_action

    def save_weights_base64(self) -> str:
        ih_weights = self.fc1.weight.data.cpu().numpy().astype(np.float32)
        h_biases = self.fc1.bias.data.cpu().numpy().astype(np.float32)
        ho_weights = self.fc2.weight.data.cpu().numpy().squeeze().astype(np.float32)
        o_bias = self.fc2.bias.data.cpu().numpy().astype(np.float32)[0]
        
        binary = b''
        binary += ih_weights.tobytes()
        binary += h_biases.tobytes()
        binary += ho_weights.tobytes()
        binary += struct.pack('f', o_bias)

        return base64.b64encode(binary).decode('ascii')
