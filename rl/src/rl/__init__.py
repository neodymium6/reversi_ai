import torch
from rl.agents.dense import DenseAgent, DenseAgentConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
HIDDEN_SIZE = 256

train_config = DenseAgentConfig(
        memory_size=int(1e5),
        hidden_size=HIDDEN_SIZE,
        batch_size=BATCH_SIZE,
        board_batch_size=256,
        device=DEVICE,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=10,
        lr=1e-4,
        gamma=0.99,
        n_episodes=160000,
        episodes_per_optimize=16,
        model_path="dense_agent.pth",
        verbose=True,
)

vs_config = DenseAgentConfig(
    memory_size=10000,
    hidden_size=HIDDEN_SIZE,
    batch_size=BATCH_SIZE,
    board_batch_size=128,
    device=DEVICE,
    eps_start=0.0,
    eps_end=0.0,
    eps_decay=10,
    lr=1e-4,
    gamma=0.99,
    n_episodes=1000,
    episodes_per_optimize=16,
    verbose=False,
)

def train():
    agent = DenseAgent(train_config)
    agent.train()

def vs_random():
    agent = DenseAgent(vs_config)
    agent.load("dense_agent.pth")
    win_rate = agent.vs_random(1000)
    print(f"Win rate: {win_rate}")

def vs_alpha_beta():
    agent = DenseAgent(vs_config)
    agent.load("dense_agent.pth")
    win_rate = agent.thunder_vs_alpha_beta(1000)
    print(f"Win rate: {win_rate}")
