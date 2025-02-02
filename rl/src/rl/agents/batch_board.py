from rust_reversi import Board
from typing import List, Tuple

class BatchBoard:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.boards: List[Board] = [Board() for _ in range(batch_size)]
        self.finished: List[bool] = [False for _ in range(batch_size)]

    def get_boards(self) -> List[Board]:
        return [board.clone() for board in self.boards]

    def do_move(self, moves: List[int]) -> Tuple[List[Board], List[float]]:
        # list for self.boards that are not game over
        new_boards = [None] * self.batch_size
        # return new_boards, rewards
        next_boards = []
        rewards = []
        for i, board, move in zip(range(self.batch_size), self.boards, moves):
            if move == 64:
                board.do_pass()
            else:
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
                # pad new board for optimization with same batch size
                new_boards[i] = Board()
                self.finished[i] = True
            else:
                rewards.append(0.0)
                new_boards[i] = board.clone()
        self.boards = new_boards
        return next_boards, rewards
    
    def do_random_move(self):
        new_boards = [None] * self.batch_size
        for i, board in enumerate(self.boards):
            if board.is_pass():
                board.do_pass()
            else:
                move = board.get_random_move()
                board.do_move(move)
            if board.is_game_over():
                # pad new board for optimization with same batch size
                new_boards[i] = Board()
                self.finished[i] = True
            else:
                new_boards[i] = board.clone()
        self.boards = new_boards

    def is_game_over(self) -> bool:
        return all(self.finished)
    
    def get_piece_mean(self) -> float:
        return sum([board.piece_sum() for board in self.boards]) / len(self.boards)
