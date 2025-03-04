import pathlib
from typing import List, Tuple
from rust_reversi import Board, Turn
import tqdm
import numpy as np
import h5py

EGAROUCID_PATH = "egaroucid"
H5_PATH = "egaroucid.h5"
DATA_PATH = pathlib.Path(EGAROUCID_PATH) / "0001_egaroucid_7_5_1_lv17"

def main() -> None:
    file_nums = range(0, 26)
    file_names = [f"{str(num).zfill(7)}.txt" for num in file_nums]
    file_paths = [DATA_PATH / file_name for file_name in file_names]
    for path in file_paths:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
    print("Loading boards...")
    all_boards = []
    for path in tqdm.tqdm(file_paths):
        boards = file_to_boards(path)
        all_boards += boards
    print(f"Loaded {len(all_boards)} boards.")
    print("Converting boards...")
    all_boards = [(board.get_board()[0], board.get_board()[1], score) for board, score in tqdm.tqdm(all_boards)]
    all_boards = np.array(all_boards, dtype=np.dtype([
        ("player_board", np.uint64),
        ("opponent_board", np.uint64),
        ("score", np.int32),
    ]))
    print("Removing duplicates...")
    all_boards = np.unique(all_boards)
    print(f"Number of unique board states: {len(all_boards)}")
    print("Saving boards...")
    with h5py.File(pathlib.Path(EGAROUCID_PATH) / H5_PATH, "w") as f:
        f.create_dataset("data", data=all_boards)


def file_to_boards(file_path: pathlib.Path) -> List[Tuple[Board, int]]:
    boards = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            board_str = line.split(" ")[0]
            score = int(line.split(" ")[1])
            board = Board()
            board.set_board_str(board_str, Turn.BLACK)
            boards.append((board, score))
    return boards

