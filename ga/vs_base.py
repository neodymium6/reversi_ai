from rust_reversi import Arena
import sys

BASE_MATRIX_PLAYER = "players/matrix_player.py"
MATRIX_PLAYER = "players/tmp_matrix_player.py"

DEPTH = 5

N_GAMES = 1000


def main():
    python = sys.executable
    piece_player = [python, BASE_MATRIX_PLAYER, str(DEPTH)]
    matrix_player = [python, MATRIX_PLAYER, str(DEPTH)]
    arena = Arena(piece_player, matrix_player)
    arena.play_n(N_GAMES)

    p1_win, p2_win, draw = arena.get_stats()
    print(f"base matrix player wins: {p1_win}")
    print(f"matrix player wins: {p2_win}")
    print(f"draw: {draw}")


if __name__ == "__main__":
    main()
