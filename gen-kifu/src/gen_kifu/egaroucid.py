import pathlib
from typing import List, Tuple
from rust_reversi import Board, Turn
import tqdm
import numpy as np
import h5py

EGAROUCID_PATH = "egaroucid"
H5_PATH = "egaroucid_augmented.h5"
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
    print("Rotating boards...")
    rotated_boards = []
    for player_board, opponent_board, score in tqdm.tqdm(all_boards):
        rotated_boards.append((player_board, opponent_board, score))
        rotated_boards.append((opponent_board, player_board, -score))
        rotate_90_p = rotate_right_90(player_board)
        rotate_90_o = rotate_right_90(opponent_board)
        rotated_boards.append((rotate_90_p, rotate_90_o, score))
        rotated_boards.append((rotate_90_o, rotate_90_p, -score))
        rotate_180_p = rotate_right_90(rotate_90_p)
        rotate_180_o = rotate_right_90(rotate_90_o)
        rotated_boards.append((rotate_180_p, rotate_180_o, score))
        rotated_boards.append((rotate_180_o, rotate_180_p, -score))
        rotate_270_p = rotate_right_90(rotate_180_p)
        rotate_270_o = rotate_right_90(rotate_180_o)
        rotated_boards.append((rotate_270_p, rotate_270_o, score))
        rotated_boards.append((rotate_270_o, rotate_270_p, -score))
        flip_h_p = flip_horizontal(player_board)
        flip_h_o = flip_horizontal(opponent_board)
        rotated_boards.append((flip_h_p, flip_h_o, score))
        rotated_boards.append((flip_h_o, flip_h_p, -score))
        flip_h_r90_p = flip_horizontal(rotate_90_p)
        flip_h_r90_o = flip_horizontal(rotate_90_o)
        rotated_boards.append((flip_h_r90_p, flip_h_r90_o, score))
        rotated_boards.append((flip_h_r90_o, flip_h_r90_p, -score))
        flip_h_r180_p = flip_horizontal(rotate_180_p)
        flip_h_r180_o = flip_horizontal(rotate_180_o)
        rotated_boards.append((flip_h_r180_p, flip_h_r180_o, score))
        rotated_boards.append((flip_h_r180_o, flip_h_r180_p, -score))
        flip_h_r270_p = flip_horizontal(rotate_270_p)
        flip_h_r270_o = flip_horizontal(rotate_270_o)
        rotated_boards.append((flip_h_r270_p, flip_h_r270_o, score))
        rotated_boards.append((flip_h_r270_o, flip_h_r270_p, -score))

    all_boards += rotated_boards
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

def rotate_right_90(x: int) -> int:
    res: int = 0
    for i in range(8):
        for j in range(8):
            bit = 1 << (63 - (8 * i + j)) & x
            if bit != 0:
                res |= 1 << (63 - (8 * j + 7 - i))
    return res

def flip_horizontal(x: int) -> int:
    res: int = 0
    res |= (x & 0x0101010101010101) << 7
    res |= (x & 0x0202020202020202) << 5
    res |= (x & 0x0404040404040404) << 3
    res |= (x & 0x0808080808080808) << 1
    res |= (x & 0x1010101010101010) >> 1
    res |= (x & 0x2020202020202020) >> 3
    res |= (x & 0x4040404040404040) >> 5
    res |= (x & 0x8080808080808080) >> 7
    return res
