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
import os
import uuid
import random
import math
import functools
from concurrent.futures import ProcessPoolExecutor
torch.multiprocessing.set_sharing_strategy('file_system')

DATA_PATH = "egaroucid_augmented.h5"
LOSS_PLOT_PATH = "loss.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pth"

MAX_DATA = int(1e6) * 10
BATCH_SIZE = 2048
LR = 5e-3
WEIGHT_DECAY = 1e-8
N_EPOCHS = 100
HIDDEN_SIZE = 64
NUM_WORKERS = 20

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

def board_to_input(board: Board) -> np.ndarray:
    board_tensor = DenseNet.board_to_input(board)
    return board_tensor.numpy()

class ReversiDataset(torch.utils.data.IterableDataset):
    def __init__(self, X: List[Tuple[Board, int]], chunk_size: int = int(3e5), shuffle: bool = True, preprocess_workers: int = 1):
        self.X = X
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.temp_dir = "tmp"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.scores = torch.tensor([x[1] for x in X], dtype=torch.float32)

        sample_tensor = torch.from_numpy(board_to_input(X[0][0]))
        self.tensor_shape = sample_tensor.shape
        self.mmap_path = os.path.join(self.temp_dir, f"tensor_cache_{uuid.uuid4()}.dat")

        total_shape = (len(X), *self.tensor_shape)
        total_size = np.prod(total_shape)
        print(f"Creating memmap of shape {total_shape} ({total_size * 4 / (1024**3):.2f} GB)")

        self.mmap_tensors = np.memmap(
            self.mmap_path,
            dtype=np.float32,
            mode="w+",
            shape=total_shape,
        )

        print(f"Computing and storing board representations using {preprocess_workers} workers")
        preprocess_minibatch_size = len(X) // preprocess_workers
        preprocess_minibatch_size = max(preprocess_minibatch_size, int(1e4))
        preprocess_minibatch_size = min(preprocess_minibatch_size, int(1e6))
        chunk_indices = [(i, min(i + preprocess_minibatch_size, len(X))) for i in range(0, len(X), preprocess_minibatch_size)]
        # convert to (player_board, opponent_board) for pickle-able
        po = [(b.get_board()[0], b.get_board()[1]) for b, _s in X]
        preprocess_func = functools.partial(
            self._preprocess_chunk,
            X=po,
            temp_dir=self.temp_dir,
            board_to_input_func=board_to_input,
        )
        with ProcessPoolExecutor(max_workers=preprocess_workers) as executor:
            futures = [
                executor.submit(preprocess_func, chunk_start, chunk_end)
                for chunk_start, chunk_end in chunk_indices
            ]
            for future in tqdm.tqdm(futures, desc="Collecting results", leave=False):
                chunk_start, chunk_end, data = future.result()
                self.mmap_tensors[chunk_start:chunk_end] = data
                self.mmap_tensors.flush()
        print(f"Completed preprocessing. Data stored at {self.mmap_path}")

        del self.mmap_tensors
        self.mmap_tensors = np.memmap(
            self.mmap_path,
            dtype=np.float32,
            mode='r',
            shape=total_shape
        )
    
    def __len__(self) -> int:
        return len(self.X)

    @staticmethod
    def _preprocess_chunk(
        chunk_start: int,
        chunk_end: int,
        X: List[Tuple[int, int]],
        temp_dir: str,
        board_to_input_func,
    ) -> Tuple[int, int, np.ndarray]:
        temp_path = os.path.join(temp_dir, f"temp_chunk_{chunk_start}_{chunk_end}_{uuid.uuid4()}.dat")
        chunk_size = chunk_end - chunk_start
        board = Board()
        board.set_board(X[0][0], X[0][1], Turn.BLACK)
        sample_shape = board_to_input_func(board).shape
        temp_memmap = np.memmap(
            temp_path,
            dtype=np.float32,
            mode="w+",
            shape=(chunk_size, *sample_shape)
        )
        try:
            for i, idx in enumerate(range(chunk_start, chunk_end)):
                board = Board()
                board.set_board(X[idx][0], X[idx][1], Turn.BLACK)
                temp_memmap[i] = board_to_input_func(board)
            temp_memmap.flush()
            result = temp_memmap.copy()
            return chunk_start, chunk_end, result
        finally:
            del temp_memmap
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_path}: {e}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_idx = 0
            end_idx = len(self.X)
        else:
            per_worker = int(math.ceil(len(self.X) / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.X))
        indices = list(range(start_idx, end_idx))
        if self.shuffle:
            random.shuffle(indices)

        for chunk_start in range(0, len(indices), self.chunk_size):
            chunk_indices = indices[chunk_start:chunk_start + self.chunk_size]
            memory_tensor = torch.from_numpy(self.mmap_tensors[chunk_indices].copy())
            for mem_idx, idx in enumerate(chunk_indices):
                tensor = memory_tensor[mem_idx]
                score = self.scores[idx]
                yield tensor, score

    def __del__(self):
        if hasattr(self, 'mmap_tensors'):
            del self.mmap_tensors
        if hasattr(self, 'mmap_path') and os.path.exists(self.mmap_path):
            try:
                os.remove(self.mmap_path)
                print(f"Cleaned up temporary file: {self.mmap_path}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {self.mmap_path}: {e}")

def main() -> None:
    print("Loading data...")
    data = load_data()

    net = DenseNet(HIDDEN_SIZE)
    net.to(DEVICE)

    data_train, data_test = train_test_split(data, test_size=0.1, shuffle=True)
    train_dataset = ReversiDataset(
        data_train,
        net,
        preprocess_workers=NUM_WORKERS,
    )
    test_dataset = ReversiDataset(
        data_test,
        net,
        shuffle=False,
        preprocess_workers=NUM_WORKERS,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=1000,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=100,
    )

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
                plot_loss(epoch, train_losses, test_losses, lrs)

                n_games = 100
                random_win_rate = vs_random(n_games, net)
                mcts_win_rate = vs_mcts(n_games, net)
                alpha_beta_win_rate = vs_alpha_beta(n_games, net)
                epoch_pb.write(f"Epoch {epoch:{len(str(N_EPOCHS))}d}: Win rate vs random: {random_win_rate:.4f}, vs MCTS: {mcts_win_rate:.4f}, vs alpha beta: {alpha_beta_win_rate:.4f}")
        lrs.append(lr_scheduler.get_last_lr()[0])
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
