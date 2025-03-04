import random
from rust_reversi import AlphaBetaSearch, Board, MatrixEvaluator, MctsSearch, Turn
from supervised_learning.models import ReversiNet


def vs_random(n_games: int, net: ReversiNet) -> float:
    print("Vs Random")
    net.eval()
    def two_game():
        win_count = 0
        # net is black
        board = Board()
        while not board.is_game_over():
            if board.is_pass():
                board.do_pass()
                continue
            if board.get_turn() == Turn.BLACK:
                action = net.get_action(board)
            else:
                action = board.get_random_move()
            board.do_move(action)
        if board.is_black_win():
            win_count += 1
        # net is white
        board = Board()
        while not board.is_game_over():
            if board.is_pass():
                board.do_pass()
                continue
            if board.get_turn() == Turn.WHITE:
                action = net.get_action(board)
            else:
                action = board.get_random_move()
            board.do_move(action)
        if board.is_white_win():
            win_count += 1
        return win_count

    win_count = 0
    for _ in range(n_games // 2):
        win_count += two_game()
    win_rate = win_count / n_games
    return win_rate

def vs_mcts(n_games: int, net: ReversiNet) -> float:
    print("Vs MCTS")
    net.eval()
    search = MctsSearch(100, 1.0, 3)
    def two_game():
        win_count = 0
        # net is black
        board = Board()
        while not board.is_game_over():
            if board.is_pass():
                board.do_pass()
                continue
            if board.get_turn() == Turn.BLACK:
                action = net.get_action(board)
            else:
                action = search.get_move(board)
            board.do_move(action)
        if board.is_black_win():
            win_count += 1
        # net is white
        board = Board()
        while not board.is_game_over():
            if board.is_pass():
                board.do_pass()
                continue
            if board.get_turn() == Turn.WHITE:
                action = net.get_action(board)
            else:
                action = search.get_move(board)
            board.do_move(action)
        if board.is_white_win():
            win_count += 1
        return win_count

    win_count = 0
    for _ in range(n_games // 2):
        win_count += two_game()
    win_rate = win_count / n_games
    return win_rate

def vs_alpha_beta(n_games: int, net: ReversiNet, epsilon: float = 0.1) -> float:
    print("Vs AlphaBeta")
    net.eval()
    def two_game():
        evaluator = MatrixEvaluator([
            [ 40,   1,  4,  0,  0,  4,   1, 40],
            [  1, -12, -8, -6, -6, -8, -12,  1],
            [  4,  -8, -1,  0,  0, -1,  -8,  4],
            [  0,  -6,  0,  0,  0,  0,  -6,  0],
            [  0,  -6,  0,  0,  0,  0,  -6,  0],
            [  4,  -8, -1,  0,  0, -1,  -8,  4],
            [  1, -12, -8, -6, -6, -8, -12,  1],
            [ 40,   1,  4,  0,  0,  4,   1, 40],
        ])
        search = AlphaBetaSearch(evaluator, 4, 1 << 10)
        win_count = 0
        # agent is black
        board = Board()
        while not board.is_game_over():
            if board.is_pass():
                board.do_pass()
                continue
            if random.random() < epsilon:
                action = board.get_random_move()
                board.do_move(action)
                continue
            if board.get_turn() == Turn.BLACK:
                action = net.get_action(board)
            else:
                action = search.get_move(board)
            board.do_move(action)
        if board.is_black_win():
            win_count += 1
        # agent is white
        board = Board()
        while not board.is_game_over():
            if board.is_pass():
                board.do_pass()
                continue
            if random.random() < epsilon:
                action = board.get_random_move()
                board.do_move(action)
                continue
            if board.get_turn() == Turn.WHITE:
                action = net.get_action(board)
            else:
                action = search.get_move(board)
            board.do_move(action)
        if board.is_white_win():
            win_count += 1
        return win_count

    win_count = 0
    for _ in range(n_games // 2):
        win_count += two_game()
    win_rate = win_count / n_games
    return win_rate
