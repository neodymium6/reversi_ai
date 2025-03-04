import h5py
from rust_reversi import Board, Turn
from typing import List, Tuple
import tqdm
import numpy as np
from supervised_learning.models.dense import DenseNet
import torch

DATA_PATH = "egaroucid.h5"
MAX_DATA = int(1e6)
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

def main() -> None:
    print("Loading data...")
    data = load_data()

    net = DenseNet(128)
    net.to(DEVICE)
    some_board = data[0][0]
    some_input = net.board_to_input(some_board).to(DEVICE)
    print(f"{net(some_input)=}")
    print(f"{net.get_action(some_board)=}")
    print(f"{some_board.get_legal_moves_vec()=}")
