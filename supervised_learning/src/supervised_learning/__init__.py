import h5py
from rust_reversi import Board, Turn
from typing import List, Tuple
import tqdm
import numpy as np
from supervised_learning.models.dense import DenseNet
from supervised_learning.models import ReversiNet
import torch
from supervised_learning.vs import vs_random, vs_mcts, vs_alpha_beta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_PATH = "egaroucid_augmented.h5"
LOSS_PLOT_PATH = "loss.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pth"

MAX_DATA = int(1e6)
BATCH_SIZE = 2048
LR = 5e-3
WEIGHT_DECAY = 1e-4
N_EPOCHS = 100
HIDDEN_SIZE = 64

def load_data() -> List[Tuple[Board, int]]:
    loaded_data: List[Tuple[Board, int]] = []
    with h5py.File(DATA_PATH, "r") as f:
        all_data = f["data"][:]
        if all_data.shape[0] > MAX_DATA:
            print(f"Data size is too large, truncate to {MAX_DATA}")
            print(f"Using {MAX_DATA / all_data.shape[0] * 100:.2f}% of data")
            all_data = np.random.choice(all_data, MAX_DATA, replace=False)
        else:
            print(f"Data size: {all_data.shape[0]}")
        for data in tqdm.tqdm(all_data):
            player_board = data[0]
            opponent_board = data[1]
            board = Board()
            board.set_board(player_board, opponent_board, Turn.BLACK)
            loaded_data.append((board, data[2]))
    return loaded_data

class ReversiDataset(torch.utils.data.Dataset):
    def __init__(self, X: List[Tuple[Board, int]], net: ReversiNet):
        self.X = X
        self.net = net
        print("Initializing ReversiDataset...")
        self.scores = torch.tensor([x[1] for x in X], dtype=torch.float32)
        self.board_tensors = torch.stack([self._board_to_input(x[0]) for x in tqdm.tqdm(X)])

    def _board_to_input(self, board: Board) -> torch.Tensor:
        return self.net.board_to_input(board)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.board_tensors[idx], self.scores[idx]

def main() -> None:
    print("Loading data...")
    data = load_data()

    net = DenseNet(HIDDEN_SIZE)
    net.to(DEVICE)

    data_train, data_test = train_test_split(data, test_size=0.1, shuffle=True)
    train_dataset = ReversiDataset(data_train, net)
    test_dataset = ReversiDataset(data_test, net)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
    )

    criterion = torch.nn.MSELoss()

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
            loss: torch.Tensor = criterion(targets, outputs.view(-1))
            total_loss += loss.item()
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
                loss = criterion(targets, outputs.view(-1))
                total_loss += loss.item()
            test_losses.append(total_loss / len(test_loader))
            lr_scheduler.step(total_loss)

            if epoch % (N_EPOCHS // 10) == 0:
                torch.save(net.state_dict(), MODEL_PATH)

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
                ax2.legend(loc='upper right')
                
                plt.title(f"Training Progress - Epoch {epoch}")
                plt.tight_layout()
                plt.savefig(LOSS_PLOT_PATH)
                plt.close(fig)

                n_games = 100
                random_win_rate = vs_random(n_games, net)
                mcts_win_rate = vs_mcts(n_games, net)
                alpha_beta_win_rate = vs_alpha_beta(n_games, net)
                epoch_pb.write(f"Epoch {epoch:{len(str(N_EPOCHS))}d}: Win rate vs random: {random_win_rate:.4f}, vs MCTS: {mcts_win_rate:.4f}, vs alpha beta: {alpha_beta_win_rate:.4f}")
        lrs.append(lr_scheduler.get_last_lr()[0])

    n_games = 500
    random_win_rate = vs_random(n_games, net)
    mcts_win_rate = vs_mcts(n_games, net)
    alpha_beta_win_rate = vs_alpha_beta(n_games, net)
    print(f"Epoch {N_EPOCHS:{len(str(N_EPOCHS))}d}: Win rate vs random: {random_win_rate:.4f}, vs MCTS: {mcts_win_rate:.4f}, vs alpha beta: {alpha_beta_win_rate:.4f}")
    torch.save(net.state_dict(), MODEL_PATH)

def export():
    net = DenseNet(HIDDEN_SIZE)
    net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    net.eval()
    base64_str = net.save_weights_base64()
    with open("weights.txt", "w") as f:
        f.write(base64_str)
