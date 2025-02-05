import copy
import torch
from rl.agents import AgentConfig, Agent
from rl.agents.net_driver import NetType
from rl.agents.net_driver.cnn import CnnConfig
from rl.agents.net_driver.dense import DenseConfig
from rl.agents.net_driver.transformer import TransformerConfig
from rl.memory import MemoryType, MemoryConfig
from rl import tuning
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
EPISODES = 120000
MEMORY_RATIO = 3.487
BATCH_BOARD_SIZE = 240

memory_config = MemoryConfig(
    memory_size=int(EPISODES * MEMORY_RATIO),
    memory_type=MemoryType.PROPORTIONAL,
    alpha=0.75456,
    beta=0.7284,
)

net_config = TransformerConfig(
    net_type=NetType.Transformer,
    patch_size=2,
    embed_dim=128,
    num_heads=4,
    num_layers=8,
    mlp_ratio=2.0,
    dropout=0.0,
)

# net_config = DenseConfig(
#     hidden_size=256,
#     net_type=NetType.Dense,
# )

train_config = AgentConfig(
    memory_config=memory_config,
    net_config=net_config,
    batch_size=BATCH_SIZE,
    board_batch_size=BATCH_BOARD_SIZE,
    n_board_init_random_moves=29,
    p_board_init_random_moves=0.8,
    device=DEVICE,
    eps_start=1.0,
    eps_end=0.077348,
    eps_decay=5,
    lr=2e-5,
    gradient_clip=1.0,
    gamma=0.991,
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
    if DEVICE == torch.device("cuda"):
        torch.set_float32_matmul_precision("high")
        print(f"Using CUDA, setting float32_matmul_precision to high")
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

def tune():
    if len(sys.argv) > 2:
        print("Usage: uv run tune [--resume (optional)]")
        raise ValueError(f"Invalid arguments: {sys.argv[1:]}")
    resume = False
    if len(sys.argv) == 2:
        if sys.argv[1] == "--resume":
            resume = True
        else:
            print("Usage: uv run tune [--resume (optional)]")
            raise ValueError(f"Invalid argument: {sys.argv[1]}")
    tuning.tune(resume=resume)
