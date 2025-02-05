from typing import List, Tuple
import pathlib
from rust_reversi import Board
import numpy as np
import h5py

WTHOR_PATH = "wthor"
H5_PATH = "unique_boards.h5"

N_GAMES = {
    1990: 1651,
    1991: 1332,
    1992: 320,
    1993: 880,
    1994: 1949,
    1995: 2429,
    1996: 2449,
    1997: 2013,
    1998: 1926,
    1999: 1817,
    2000: 2396,
    2001: 2208,
    2002: 1891,
    2003: 2172,
    2004: 4348,
    2005: 2232,
    2006: 2478,
    2007: 2942,
    2008: 4199,
    2009: 9113,
    2010: 3858,
    2011: 5423,
    2012: 5575,
    2013: 4253,
    2014: 7685,
    2015: 8077,
    2016: 7681,
    2017: 5852,
    2018: 4968,
    2019: 4343,
    2020: 3484,
    2021: 3347,
    2022: 3152,
    2023: 2986,
    2024: 1157,
}

def read_wtb(year: int) -> List[List[int]]:
    dir_path = pathlib.Path(WTHOR_PATH)
    all_moves = []
    with open(dir_path / str(year) / ("WTH_" + str(year) + ".wtb"), "rb") as f:
        _header = f.read(16)
        for _ in range(N_GAMES[year]):
            _name = f.read(2)
            _black_player = f.read(2)
            _white_player = f.read(2)
            _black_pieces = f.read(1)
            _theoretical = f.read(1)
            moves = []
            move_bs = []
            board = Board()
            for _ in range(60):
                if board.is_pass() and not board.is_game_over():
                    board.do_pass()
                move_b = int.from_bytes(f.read(1), byteorder="little")
                move_bs.append(move_b)
                move_x = move_b // 10 - 1
                move_y = move_b % 10 - 1
                move = move_x + move_y * 8
                if move_b == 0:
                    continue
                try:
                    board.do_move(move)
                except Exception:
                    print(move)
                    print(board)
                    raise
                moves.append(move)
            if len(moves) != 0:
                all_moves.append(moves)
    return all_moves

def play_moves(moves: List[int]) -> List[Tuple[int, int, str]]:
    board_history = []
    board = Board()
    for move in moves:
        player_board, opponent_board, turn = board.get_board()
        board_history.append((player_board, opponent_board, str(turn)))
        if board.is_pass():
            board.do_pass()
            player_board, opponent_board, turn = board.get_board()
            board_history.append((player_board, opponent_board, str(turn)))
        board.do_move(move)
        if board.is_game_over():
            break
    assert len(board_history) >= len(moves)
    return board_history

def main() -> None:
    years = range(1990, 2025)
    all_board_history = []
    try:
        for year in years:
            moves = read_wtb(year)
            for game in moves:
                board_history = play_moves(game)
                all_board_history += board_history
    except Exception:
        raise
    all_board_history = np.array(all_board_history, dtype=np.dtype([
        ("player_board", np.uint64),
        ("opponent_board", np.uint64),
        ("turn", "S5"),
    ]))
    all_board_history = np.unique(all_board_history, axis=0)
    print("Done reading WTB files.")
    print(f"Number of unique board states: {len(all_board_history)}")
    with h5py.File(pathlib.Path(WTHOR_PATH) / H5_PATH, "w") as f:
        f.create_dataset("unique_boards", data=all_board_history)
