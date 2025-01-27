from rust_reversi import Board, Turn
import torch
from rl.agents.dense import DenseAgent, DenseAgentConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

def train():
    config = DenseAgentConfig(
        memory_size=10000,
        hidden_size=256,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        eps_start=0.9,
        eps_end=0.05,
        lr=1e-4,
        gamma=0.99,
        n_episodes=1000,
        verbose=True,
    )
    agent = DenseAgent(config)
    agent.train()
    agent.save("dense_agent.pth")


def vs_random():
    config = DenseAgentConfig(
        memory_size=10000,
        hidden_size=256,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        eps_start=0.9,
        eps_end=0.05,
        lr=1e-4,
        gamma=0.99,
        n_episodes=1000,
        verbose=False,
    )
    agent = DenseAgent(config)
    agent.load("dense_agent.pth")
    
    def two_game():
        win_count = 0
        # agent is black
        board = Board()
        while not board.is_game_over():
            if board.is_pass():
                board.do_pass()
                continue
            _p, _o, turn = board.get_board()
            if turn == Turn.BLACK:
                action = agent.get_action(board, 1.0)
            else:
                action = board.get_random_move()
            board.do_move(action)
        if board.is_black_win():
            win_count += 1
        # agent is white
        board = Board()
        while not board.is_game_over():
            if board.is_pass():
                board.do_pass()
                continue
            _p, _o, turn = board.get_board()
            if turn == Turn.WHITE:
                action = agent.get_action(board, 1.0)
            else:
                action = board.get_random_move()
            board.do_move(action)
        if board.is_white_win():
            win_count += 1
        return win_count
    
    win_count = 0
    n_games = 1000
    for _ in range(n_games // 2):
        win_count += two_game()
    print(f"Win rate: {win_count / n_games}")
