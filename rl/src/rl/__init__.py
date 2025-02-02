import copy
import torch
from rl.agents import AgentConfig, Agent
from rl.agents.net_driver import NetType
from rl.agents.net_driver.cnn import CnnConfig
from rl.agents.net_driver.dense import DenseConfig
from rl.memory import MemoryType, MemoryConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
EPISODES = 120000

memory_config = MemoryConfig(
    memory_size=EPISODES // 5,
    memory_type=MemoryType.PROPORTIONAL,
    alpha=0.5,
    beta=0.5,
)

net_config = CnnConfig(
    num_channels=64,
    fc_hidden_size=256,
    net_type=NetType.RESNET10,
)

# net_config = DenseConfig(
#     hidden_size=256,
#     net_type=NetType.Dense,
# )

train_config = AgentConfig(
    memory_config=memory_config,
    net_config=net_config,
    batch_size=BATCH_SIZE,
    board_batch_size=240,
    n_board_init_random_moves=14,
    p_board_init_random_moves=0.8,
    device=DEVICE,
    eps_start=1.0,
    eps_end=0.03,
    eps_decay=10,
    lr=1e-5,
    gradient_clip=1.0,
    gamma=0.99,
    n_episodes=EPISODES,
    steps_per_optimize=1,
    optimize_per_target_update=1,
    verbose=True,
    model_path="cnn_agent.pth",
)

vs_config = copy.deepcopy(train_config)
vs_config["eps_start"] = 0.0
vs_config["eps_end"] = 0.0
vs_config["verbose"] = False

def train():
    train_agent = Agent(train_config)
    train_agent.train()

def vs_random():
    vs_agent = Agent(vs_config)
    vs_agent.load(vs_agent.config["model_path"])
    win_rate = vs_agent.vs_random(1000)
    print(f"Win rate: {win_rate}")

def vs_alpha_beta():
    vs_config["verbose"] = True
    vs_agent = Agent(vs_config)
    vs_agent.load(vs_agent.config["model_path"])
    win_rate = vs_agent.thunder_vs_alpha_beta(1000)
    print(f"Win rate: {win_rate}")
