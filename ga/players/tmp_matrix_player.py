import sys
from rust_reversi import Board, Turn, MatrixEvaluator, AlphaBetaSearch
import random

DEPTH = int(sys.argv[1])
EPSILON = 1e-1

EVAL_MATRIX = [
    [4, 16, -9, -1, -9, -30, 6, -9],
    [-13, 11, -1, -9, 2, -21, 9, 8],
    [12, -8, 5, -12, -7, -25, -16, 9],
    [-11, -27, -1, 5, 12, -15, -19, 9],
    [-16, 1, 10, 2, -17, 15, -2, 3],
    [8, -7, -15, 8, 11, 3, -14, -9],
    [19, 10, 5, 13, -5, 0, -18, 16],
    [23, 2, -4, 14, 8, 5, -20, 16],
]


def main():
    turn = Turn.BLACK if sys.argv[2] == "BLACK" else Turn.WHITE
    board = Board()
    evaluator = MatrixEvaluator(EVAL_MATRIX)
    search = AlphaBetaSearch(evaluator, DEPTH)

    while True:
        try:
            board_str = input().strip()

            if board_str == "ping":
                print("pong", flush=True)
                continue

            board.set_board_str(board_str, turn)
            if random.random() < EPSILON:
                move = board.get_random_move()
            else:
                move = search.get_move(board)

            print(move, flush=True)

        except Exception as e:
            print(e, file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
