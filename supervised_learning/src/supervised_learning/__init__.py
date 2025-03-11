import h5py
from typing import List, Tuple
import tqdm
import numpy as np
from supervised_learning.models.dense import DenseNet
from supervised_learning.models import ReversiNet
import torch
from supervised_learning.vs import vs_random, vs_mcts, vs_alpha_beta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from supervised_learning.reversi_dataset import ReversiDataset
from torch_optimizer import Lookahead
torch.multiprocessing.set_sharing_strategy('file_system')

DATA_PATH = "egaroucid_augmented.h5"
LOSS_PLOT_PATH = "loss.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pth"

MAX_DATA = int(1e6) * 20 * 3
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 5e-7
N_EPOCHS = 100
HIDDEN_SIZE = 64
PREPROCESS_WORKERS = 16
NUM_WORKERS = 4

def load_data() -> List[Tuple[int, int, int]]:
    loaded_data: List[Tuple[int, int, int]] = []
    with h5py.File(DATA_PATH, "r") as f:
        all_data = f["data"][:]
        if all_data.shape[0] > MAX_DATA:
            print(f"Data size is too large, truncate to {MAX_DATA}")
            print(f"Using {MAX_DATA / all_data.shape[0] * 100:.2f}% of data")
            all_data = np.random.choice(all_data, MAX_DATA, replace=False)
        else:
            print(f"Data size: {all_data.shape[0]}")
        for data in tqdm.tqdm(all_data, desc="Loading data", leave=False):
            player_board = data[0]
            opponent_board = data[1]
            score = data[2]
            loaded_data.append((player_board, opponent_board, score))
    return loaded_data

def main() -> None:
    print("Loading data...")
    data = load_data()

    net = DenseNet(HIDDEN_SIZE)
    net.to(DEVICE)

    data_train, data_test = train_test_split(data, test_size=0.1, shuffle=True)
    train_dataset = ReversiDataset(
        data_train,
        preprocess_workers=PREPROCESS_WORKERS,
    )
    test_dataset = ReversiDataset(
        data_test,
        shuffle=False,
        preprocess_workers=PREPROCESS_WORKERS,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=50,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=10,
    )

    base_optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
    )

    criterion = torch.nn.SmoothL1Loss()

    train_losses = []
    test_losses = []
    lrs = []
    epoch_pb = tqdm.tqdm(range(N_EPOCHS))
    for epoch in epoch_pb:
        net.train()
        total_loss = 0.0
        train_pb = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for inputs, targets in train_pb:
            inputs: torch.Tensor
            targets: torch.Tensor
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss: torch.Tensor = criterion(outputs.view(-1), targets)
            total_loss += loss.item()
            train_pb.set_description(f"Epoch {epoch} - Loss: {loss.item():.2f}")
            loss.backward()
            optimizer.step()
        train_losses.append(total_loss / len(train_loader))
        net.eval()
        with torch.no_grad():
            total_loss = 0.0
            for inputs, targets in test_loader:
                inputs: torch.Tensor
                targets: torch.Tensor
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs: torch.Tensor = net(inputs)
                loss = criterion(outputs.view(-1), targets)
                total_loss += loss.item()
            test_losses.append(total_loss / len(test_loader))
            lr_scheduler.step(total_loss)
            lrs.append(lr_scheduler.get_last_lr()[0])
            plot_loss(epoch, train_losses, test_losses, lrs)

            if epoch % (N_EPOCHS // 10) == 0:
                torch.save(net.state_dict(), MODEL_PATH)

                n_games = 100
                random_win_rate = vs_random(n_games, net)
                mcts_win_rate = vs_mcts(n_games, net)
                alpha_beta_win_rate = vs_alpha_beta(n_games, net)
                epoch_pb.write(f"Epoch {epoch:{len(str(N_EPOCHS))}d}: Win rate vs random: {random_win_rate:.4f}, vs MCTS: {mcts_win_rate:.4f}, vs alpha beta: {alpha_beta_win_rate:.4f}")
    torch.save(net.state_dict(), MODEL_PATH)
    plot_loss(N_EPOCHS, train_losses, test_losses, lrs)

    n_games = 500
    random_win_rate = vs_random(n_games, net)
    mcts_win_rate = vs_mcts(n_games, net)
    alpha_beta_win_rate = vs_alpha_beta(n_games, net)
    print(f"Epoch {N_EPOCHS:{len(str(N_EPOCHS))}d}: Win rate vs random: {random_win_rate:.4f}, vs MCTS: {mcts_win_rate:.4f}, vs alpha beta: {alpha_beta_win_rate:.4f}")

def plot_loss(epoch: int, train_losses: List[float], test_losses: List[float], lrs: List[float]) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = 'blue'
    color2 = 'orange'
    ax1.plot(train_losses, label="train", color=color1)
    ax1.plot(test_losses, label="test", color=color2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color3 = 'red'
    ax2.plot(lrs, label="learning rate", color=color3, linestyle='--')
    ax2.set_ylabel("Learning Rate")
    ax2.set_yscale("log")
    ax2.legend(loc='upper right')
    
    plt.title(f"Training Progress - Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close(fig)

def export():
    net = DenseNet(HIDDEN_SIZE)
    net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    net.eval()
    base64_str = net.save_weights_base64()
    with open("weights.txt", "w") as f:
        f.write(base64_str)
