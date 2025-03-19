import torch
from rust_reversi import Board
from supervised_learning.models import ReversiNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNet(ReversiNet):
    def __init__(
            self,
            c1: int = 16,
            c2: int = 16,
            c3: int = 16,
        ):
        super(ConvNet, self).__init__()
        # 2x8x8 -> c1x6x6
        self.conv1 = torch.nn.Conv2d(
            in_channels=2,
            out_channels=c1,
            kernel_size=3,
            padding=0,
        )
        # c1x6x6 -> c2x4x4
        self.conv2 = torch.nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=3,
            padding=0,
        )
        # c2x4x4 -> c3x2x2
        self.conv3 = torch.nn.Conv2d(
            in_channels=c2,
            out_channels=c3,
            kernel_size=3,
            padding=0,
        )
        self.fc1_input_size = c3 * 2 * 2
        self.fc1 = torch.nn.Linear(self.fc1_input_size, 1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = x.view(-1, self.fc1_input_size)
        x = self.fc1(x)
        return x

    @staticmethod
    def board_to_input(board: Board) -> torch.Tensor:
        board_matrix = board.get_board_matrix()[0:2]
        input_tensor = torch.tensor(board_matrix, dtype=torch.float32)
        return input_tensor

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
        values = -values
        _best_value, best_index = torch.max(values, 0)
        best_action = legal_actions[best_index.item()]
        return best_action

    def save_weights_base64(self) -> str:
        pass
