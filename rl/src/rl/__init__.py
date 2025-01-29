import copy
import torch
from rl.agents.cnn import CnnAgent, CnnAgentConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256

train_config = CnnAgentConfig(
        memory_size=int(1e5),
        batch_size=BATCH_SIZE,
        board_batch_size=256,
        device=DEVICE,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=10,
        lr=1e-5,
        gradient_clip=0.5,
        gamma=0.99,
        n_episodes=160000,
        episodes_per_optimize=16,
        verbose=True,

        num_channels=128,
        fc_hidden_size=128,
        model_path="cnn_agent.pth",
)

vs_config = copy.deepcopy(train_config)
vs_config["eps_start"] = 0.0
vs_config["eps_end"] = 0.0
vs_config["verbose"] = False

train_agent = CnnAgent(train_config)
vs_agent = CnnAgent(vs_config)

def train():
    train_agent.train()
    train_agent.plot()

def vs_random():
    vs_agent.load(vs_agent.config["model_path"])
    win_rate = vs_agent.vs_random(1000)
    print(f"Win rate: {win_rate}")

def vs_alpha_beta():
    vs_agent.load(vs_agent.config["model_path"])
    win_rate = vs_agent.vs_alpha_beta(1000)
    print(f"Win rate: {win_rate}")
