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
