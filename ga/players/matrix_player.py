import sys
from rust_reversi import Board, Turn, MatrixEvaluator, AlphaBetaSearch
import random

DEPTH = int(sys.argv[1])
EPSILON = 5e-2

EVAL_MATRIX = [
    [15, 0, -1, 0, 9, -1, -1, 15],
    [-3, -7, -1, 0, 0, -1, -7, -9],
    [-1, -1, 2, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, -9, 0, 0, 0, 1],
    [7, -1, -9, 0, 0, 0, -1, 5],
    [0, -5, -1, 0, -5, -1, -9, 0],
    [15, -3, -1, 0, 0, 6, 0, 15],
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
