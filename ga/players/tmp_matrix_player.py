import sys
from rust_reversi import Board, Turn, MatrixEvaluator, AlphaBetaSearch
import random

DEPTH = int(sys.argv[1])
EPSILON = 1e-1

EVAL_MATRIX = [
    [-46, 0, 37, 42, 45, -35, -32, 12],
    [30, 17, 12, -21, -29, -34, -46, -4],
    [-24, -22, 37, -44, -17, -49, 26, 15],
    [49, 3, -27, 8, -41, -49, 40, -45],
    [-32, -23, 4, 0, -4, -2, 10, 28],
    [-15, 40, 39, -45, -4, -1, -44, -2],
    [48, 6, -30, 32, 20, 2, 22, -35],
    [-18, 21, 43, 33, 24, 30, -23, 11],
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
