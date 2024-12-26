import sys
from rust_reversi import Board, Turn, MatrixEvaluator, AlphaBetaSearch
import random

DEPTH = int(sys.argv[1])
EPSILON = 1e-1

EVAL_MATRIX = [
    [62, 14, 22, -6, 18, 9, 8, 88],
    [-11, -21, -13, -10, 2, 15, -18, -6],
    [8, -3, -18, -18, 0, 3, 1, 16],
    [1, -8, -17, -9, -4, -1, -30, 19],
    [28, 7, -7, 3, -1, 1, -5, 16],
    [12, 3, 8, 17, -13, -16, 0, 0],
    [9, -30, -4, -8, 5, -10, -21, -20],
    [58, 30, 2, 4, -17, 15, 10, 45],
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
