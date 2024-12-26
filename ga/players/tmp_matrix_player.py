import sys
from rust_reversi import Board, Turn, MatrixEvaluator, AlphaBetaSearch
import random

DEPTH = int(sys.argv[1])
EPSILON = 5e-2

EVAL_MATRIX = [
    [6, -7, 1, 7, 0, 1, -4, 13],
    [0, -3, -5, 0, 0, -5, -5, 0],
    [1, -5, 0, 0, 0, 0, -5, 1],
    [0, 0, 0, -1, -1, 0, 0, 0],
    [0, 0, -2, -8, -1, 0, 0, 0],
    [1, -5, 0, 4, 0, 0, -5, 1],
    [-3, 1, -5, 0, -9, -5, -3, 0],
    [6, 2, 1, 0, -9, 1, 0, 12],
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
