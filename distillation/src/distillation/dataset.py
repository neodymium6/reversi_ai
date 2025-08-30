import numpy as np
import torch
import tqdm
from rust_reversi import Board, Turn
from distillation.models import ReversiNet

class DistillationDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            X: np.ndarray,
            teacher_net: ReversiNet,
            student_net: ReversiNet,
            device: torch.device,
            initializing_batch_size: int,
            scaling_factor: float = 1.0,
        ):
        self.X = X
        self.teacher_net = teacher_net
        self.student_net = student_net
        self.device = device
        self.teacher_v = torch.zeros(len(X), 1, dtype=torch.float32)
        idx = 0
        pb = tqdm.tqdm(total=len(X) // initializing_batch_size, desc="Initializing Teacher V")
        while idx + initializing_batch_size < len(X):
            teacher_input = torch.stack([self._x2teacher_input(x) for x in X[idx:idx+initializing_batch_size]], dim=0).to(self.device)
            legal_actions = torch.stack([self._legal_actions(x) for x in X[idx:idx+initializing_batch_size]], dim=0)
            with torch.no_grad():
                teacher_output = teacher_net(teacher_input)
                teacher_output = teacher_output.masked_fill_(~legal_actions, -1e9)
                teacher_v = torch.max(teacher_output, dim=1, keepdim=True).values
                self.teacher_v[idx:idx+initializing_batch_size] = teacher_v * scaling_factor
            idx += initializing_batch_size
            pb.update(1)
        pb.close()
        teacher_input = torch.stack([self._x2teacher_input(x) for x in X[idx:]], dim=0).to(self.device)
        legal_actions = torch.stack([self._legal_actions(x) for x in X[idx:]], dim=0)
        with torch.no_grad():
            teacher_output = teacher_net(teacher_input)
            teacher_output = teacher_output.masked_fill_(~legal_actions, -1e9)
            teacher_v = torch.max(teacher_output, dim=1, keepdim=True).values
            self.teacher_v[idx:] = teacher_v
        self.teacher_v = self.teacher_v.to(self.device)
        self.student_input = torch.stack([self._x2student_input(x) for x in tqdm.tqdm(X, desc="Initializing Student Input")], dim=0)

    def __len__(self):
        return len(self.X)
    
    def _x2teacher_input(self, x: np.void) -> torch.Tensor:
        return self.teacher_net.x2input(x)
    
    def _x2student_input(self, x: np.void) -> torch.Tensor:
        return self.student_net.x2input(x)

    def _legal_actions(self, x: np.void) -> torch.Tensor:
        player_board = x["player_board"]
        opponent_board = x["opponent_board"]
        turn_str = x["turn"]
        if turn_str == b'Black':
            turn = Turn.BLACK
        else:
            turn = Turn.WHITE
        board = Board()
        board.set_board(player_board, opponent_board, turn)
        legal_actions = torch.tensor(
            board.get_legal_moves_tf() + [board.is_pass()],
            dtype=torch.bool,
            device=self.device,
        )
        return legal_actions

    def __getitem__(self, idx):
        return (self.student_input[idx], self.teacher_v[idx])
