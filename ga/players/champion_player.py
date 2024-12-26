import sys
from rust_reversi import Board, Turn, MatrixEvaluator, AlphaBetaSearch
import random

DEPTH = int(sys.argv[1])
EPSILON = 1e-1

EVAL_MATRIX = [
    [73, -10, 7, 22, 31, 9, -1, 52],
    [5, -22, 2, 2, -8, -6, -31, -4],
    [3, -10, 0, 0, 9, 15, -5, 13],
    [12, -3, -2, 1, 2, -18, 10, -1],
    [-1, -2, -12, -3, -7, -12, -5, -6],
    [22, -21, 10, 4, 0, 7, 9, -3],
    [-5, -10, -13, 16, -21, 2, -28, -11],
    [61, -22, 15, 10, 19, 16, -7, 54],
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
