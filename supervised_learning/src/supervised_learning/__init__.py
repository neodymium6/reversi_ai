import h5py
from rust_reversi import Board, Turn
from typing import List, Tuple
import tqdm
import numpy as np
from supervised_learning.models.dense import DenseNet
from supervised_learning.models import ReversiNet
import torch
from supervised_learning.vs import vs_random, vs_mcts, vs_alpha_beta

DATA_PATH = "egaroucid.h5"
MAX_DATA = int(1e4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data() -> List[Tuple[Board, int]]:
    loaded_data: List[Tuple[Board, int]] = []
    with h5py.File(DATA_PATH, "r") as f:
        all_data = f["data"][:]
        if all_data.shape[0] > MAX_DATA:
            print(f"Data size is too large, truncate to {MAX_DATA}")
            all_data = np.random.choice(all_data, MAX_DATA, replace=False)
        for data in tqdm.tqdm(all_data):
            player_board = data[0]
            opponent_board = data[1]
            board = Board()
            board.set_board(player_board, opponent_board, Turn.BLACK)
            loaded_data.append((board, data[2]))
    return loaded_data

class ReversiDataset(torch.utils.data.Dataset):
    def __init__(self, X: List[Tuple[Board, int]], net: ReversiNet):
        self.X = X
        self.net = net
        print("Initializing ReversiDataset...")
        self.scores = torch.tensor([x[1] for x in X], dtype=torch.float32)
        self.board_tensors = torch.stack([self._board_to_input(x[0]) for x in tqdm.tqdm(X)])

    def _board_to_input(self, board: Board) -> torch.Tensor:
        return self.net.board_to_input(board)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.board_tensors[idx], self.scores[idx]

def main() -> None:
    print("Loading data...")
    data = load_data()

    net = DenseNet(128)
    net.to(DEVICE)

    dataset = ReversiDataset(data, net)
