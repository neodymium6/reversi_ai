from rust_reversi import Board, Turn
import torch

def main():
    # Start a new game
    board = Board()
    while not board.is_game_over():
        if board.is_pass():
            board.do_pass()
            continue
        # Get random move
        move = board.get_random_move()
        # Execute move
        board.do_move(move)

    # Game over
    winner = board.get_winner()
    if winner is None:
        print("Game drawn.")
    elif winner == Turn.BLACK:
        print("Black wins!")
    else:
        print("White wins!")


if __name__ == "__main__":
    main()
    print(torch.__version__)
    print(torch.cuda.is_available())
