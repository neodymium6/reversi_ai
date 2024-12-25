from rust_reversi import Arena
import sys

RANDOM_PLAYER = "players/random_player.py"
MATRIX_PLAYER = "players/matrix_player.py"
DEPTH = 4

N_GAMES = 1000


def main():
    python = sys.executable
    random_player = [python, RANDOM_PLAYER]
    matrix_player = [python, MATRIX_PLAYER, str(DEPTH)]
    arena = Arena(random_player, matrix_player)
    arena.play_n(N_GAMES)

    p1_win, p2_win, draw = arena.get_stats()
    print(f"random player wins: {p1_win}")
    print(f"matrix player wins: {p2_win}")
    print(f"draw: {draw}")


if __name__ == "__main__":
    main()
