from rust_reversi import Board

class BatchBoard:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.boards: list[Board] = [Board() for _ in range(batch_size)]

    def get_boards(self) -> list[Board]:
        return [board.clone() for board in self.boards]

    def do_move(self, moves: list[int]) -> tuple[list[Board], list[float]]:
        # list for self.boards that are not game over
        new_boards = []
        # return new_boards, rewards
        next_boards = []
        rewards = []
        for board, move in zip(self.boards, moves):
            board.do_move(move)
            next_boards.append(board.clone())
            if board.is_game_over():
                if board.is_win():
                    # turn swapped in do_move, so is_win menas the player that just moved lost
                    rewards.append(0.0)
                elif board.is_lose():
                    # turn swapped in do_move, so is_lose menas the player that just moved won
                    rewards.append(1.0)
                else:
                    # draw
                    rewards.append(0.5)
            else:
                rewards.append(0.0)
                if board.is_pass():
                    board.do_pass()
                # if the game is not over, add the board to new_boards in movable state
                new_boards.append(board.clone())
        self.boards = new_boards
        return next_boards, rewards

    def is_game_over(self) -> bool:
        return len(self.boards) == 0
