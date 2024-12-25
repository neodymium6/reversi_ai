from rust_reversi import Arena
import sys

PIECE_PLAYER = "players/piece_player.py"
MATRIX_PLAYER = "players/tmp_matrix_player.py"

PIECE_DEPTH = 4
MATRIX_DEPTH = 2

N_GAMES = 1000


def main():
    python = sys.executable
    piece_player = [python, PIECE_PLAYER, str(PIECE_DEPTH)]
    matrix_player = [python, MATRIX_PLAYER, str(MATRIX_DEPTH)]
    arena = Arena(piece_player, matrix_player)
    arena.play_n(N_GAMES)

    p1_win, p2_win, draw = arena.get_stats()
    print(f"piece player wins: {p1_win}")
    print(f"matrix player wins: {p2_win}")
    print(f"draw: {draw}")


if __name__ == "__main__":
    main()
