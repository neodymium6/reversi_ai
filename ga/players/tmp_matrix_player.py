import sys
from rust_reversi import Board, Turn, MatrixEvaluator, AlphaBetaSearch
import random

DEPTH = int(sys.argv[1])
EPSILON = 1e-1

EVAL_MATRIX = [
    [41, 3, -18, 34, -36, 16, 42, 48],
    [27, -47, -38, -52, -12, -22, 65, 13],
    [-48, -49, 27, -16, -6, 52, 6, 38],
    [-17, 38, -30, -35, 49, -50, -33, 26],
    [-21, 0, 51, 39, 46, -43, 62, -30],
    [-12, -33, 49, 19, 52, 16, -43, 24],
    [26, -3, -4, 46, -44, -5, -61, 21],
    [15, -33, 51, -2, 13, 54, 10, 37],
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
