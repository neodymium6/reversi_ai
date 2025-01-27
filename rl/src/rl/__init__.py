from rust_reversi import Board
import torch
from rl.agents.dense import DenseAgent, DenseAgentConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

def main():
    config = DenseAgentConfig(
        memory_size=10000,
        hidden_size=256,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        verbose=True,
    )
    agent = DenseAgent(config)
    board = Board()
    action = agent.get_action(board)
    print(action)
