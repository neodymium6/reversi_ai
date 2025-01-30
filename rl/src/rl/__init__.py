import copy
import torch
from rl.agents.cnn import CnnAgent, CnnAgentConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512

train_config = CnnAgentConfig(
        memory_size=int(1e5),
        batch_size=BATCH_SIZE,
        board_batch_size=240,
        device=DEVICE,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=10,
        lr=1e-6,
        gradient_clip=1.0,
        gamma=0.99,
        n_episodes=160000 * 3,
        episodes_per_optimize=16,
        episodes_per_target_update=16 * 16,
        verbose=True,

        num_channels=192,
        fc_hidden_size=256,
        model_path="cnn_agent.pth",
)

vs_config = copy.deepcopy(train_config)
vs_config["eps_start"] = 0.0
vs_config["eps_end"] = 0.0
vs_config["verbose"] = False

def train():
    train_agent = CnnAgent(train_config)
    train_agent.train()
    train_agent.plot()
def vs_random():
    vs_agent = CnnAgent(vs_config)
    vs_agent.load(vs_agent.config["model_path"])
    win_rate = vs_agent.vs_random(1000)
    print(f"Win rate: {win_rate}")

def vs_alpha_beta():
    vs_config["verbose"] = True
    vs_agent = CnnAgent(vs_config)
    vs_agent.load(vs_agent.config["model_path"])
    win_rate = vs_agent.thunder_vs_alpha_beta(1000)
    print(f"Win rate: {win_rate}")
