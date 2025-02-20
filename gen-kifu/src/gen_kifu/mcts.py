from typing import List, Tuple
from rust_reversi import Board, MctsSearch, Turn
import numpy as np
import h5py
from pathlib import Path

N_PLAYOUTS = 10000
C = 1.0
EXPAND_THRESHOLD = 10

MCTS_PATH = "mcts"
DATA_PATH = "kifu_10000.h5"

def game() -> List[Tuple[int, int, str]]:
    board_history = []
    board = Board()
    search = MctsSearch(N_PLAYOUTS, C, EXPAND_THRESHOLD)
    while not board.is_game_over():
        if board.is_pass():
            board.do_pass()
            p, o, t = board.get_board()
            board_history.append((p, o, str(t)))
        move = search.get_move(board)
        board.do_move(move)
        p, o, t = board.get_board()
        board_history.append((p, o, str(t)))
    return board_history

def main(resume: bool) -> None:
    if resume:
        print("Resuming...")
        if not (Path(MCTS_PATH) / DATA_PATH).exists():
            raise FileNotFoundError(f"File not found: {Path(MCTS_PATH) / DATA_PATH}")
        with h5py.File(Path(MCTS_PATH) / DATA_PATH, "r") as f:
            all_board_history = f["data"][:]
            all_board_history = list(all_board_history)
        print(f"Loaded {len(all_board_history)} unique board states.")
    else:
        if (Path(MCTS_PATH) / DATA_PATH).exists():
            raise FileExistsError(f"File already exists: {Path(MCTS_PATH) / DATA_PATH}")
        all_board_history = []
    print("Generating kifu...")
    print("^C to stop.")
    while True:
        try:
            board_history = game()
            all_board_history += board_history
        except KeyboardInterrupt:
            print("Stopped.")
            break
    all_board_history = np.array(all_board_history, dtype=np.dtype([
        ("player_board", np.uint64),
        ("opponent_board", np.uint64),
        ("turn", "S5"),
    ]))
    all_board_history = np.unique(all_board_history)
    print(f"Generated {len(all_board_history)} unique board states.")
    with h5py.File(Path(MCTS_PATH) / DATA_PATH, "w") as f:
        f.create_dataset("data", data=all_board_history)
