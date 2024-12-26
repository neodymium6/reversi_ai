import sys
from rust_reversi import Board, Turn, MatrixEvaluator, AlphaBetaSearch
import random

DEPTH = int(sys.argv[1])
EPSILON = 1e-1

EVAL_MATRIX = [
    [13, -9, 14, 16, 49, 15, 48, 35],
    [25, -48, -9, -33, 14, 49, 32, 8],
    [-10, -42, 20, -3, 38, 23, 29, 11],
    [-12, 24, 47, 23, -6, -50, 2, 9],
    [22, 10, -11, -49, -20, -14, 30, 13],
    [19, -19, 25, -16, -4, 0, -43, 46],
    [-6, -27, -21, -38, -40, -8, -42, 39],
    [20, 45, 28, 9, -32, 11, 15, -13],
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
