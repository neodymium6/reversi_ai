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
        eps_decay=10,
        lr=1e-4,
        gamma=0.99,
        # n_episodes=50000,
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
        eps_start=0.0,
        eps_end=0.0,
        eps_decay=10,
        lr=1e-4,
        gamma=0.99,
        n_episodes=1000,
        verbose=False,
    )
    agent = DenseAgent(config)
    agent.load("dense_agent.pth")
    win_rate = agent.vs_random(1000)
    print(f"Win rate: {win_rate}")
