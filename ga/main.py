from rust_reversi import Board, Turn


def main():
    board = Board()

    while not board.is_game_over():
        if board.is_pass():
            board.do_pass()
            continue

        move = board.get_random_move()

        board.do_move(move)

    winner = board.get_winner()
    if winner is None:
        print("Game drawn.")
    elif winner == Turn.BLACK:
        print("Black wins!")
    else:
        print("White wins!")


if __name__ == "__main__":
    main()
