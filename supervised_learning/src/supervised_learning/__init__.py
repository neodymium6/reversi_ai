import h5py
from typing import List, Tuple
import tqdm
import numpy as np
from supervised_learning.models.dense import DenseNet
import torch
from sklearn.model_selection import train_test_split
from supervised_learning.reversi_dataset import ReversiDataset
from torch_optimizer import Lookahead
from supervised_learning.losses.sign_aware import SignAwareMAE
import time
from supervised_learning.reversi_trainer import Trainer
torch.multiprocessing.set_sharing_strategy('file_system')

DATA_PATH = "egaroucid_augmented.h5"
LOSS_PLOT_PATH = "loss.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model.pth"

MAX_DATA = int(1e6) * 1
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-6
N_EPOCHS = 100
HIDDEN_SIZE = 64
PREPROCESS_WORKERS = 16
NUM_WORKERS = 4
DATA_LOADER_UPDATE_PER_EPOCH = 20
MODEL_CLASS = DenseNet

def load_data(verbose: bool) -> List[Tuple[int, int, int]]:
    if verbose:
        print("Loading data...")
    loaded_data: List[Tuple[int, int, int]] = []
    with h5py.File(DATA_PATH, "r") as f:
        all_data = f["data"][:]
        if all_data.shape[0] > MAX_DATA:
            if verbose:
                print(f"Data size is too large, truncate to {MAX_DATA}")
                print(f"Using {MAX_DATA / all_data.shape[0] * 100:.2f}% of data")
            all_data = np.random.choice(all_data, MAX_DATA, replace=False)
        else:
            if verbose:
                print(f"Data size: {all_data.shape[0]}")
        for data in tqdm.tqdm(all_data, desc="Loading data", leave=False):
            player_board = data[0]
            opponent_board = data[1]
            score = data[2]
            loaded_data.append((player_board, opponent_board, score))
    return loaded_data

def get_dataloaders(verbose=True) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    start_time = time.time()
    data = load_data(verbose)
    data_train, data_test = train_test_split(data, test_size=0.1, shuffle=True)
    del data
    train_dataset = ReversiDataset(
        X=data_train,
        model_class=MODEL_CLASS,
        preprocess_workers=PREPROCESS_WORKERS,
        verbose=verbose,
    )
    del data_train
    test_dataset = ReversiDataset(
        X=data_test,
        model_class=MODEL_CLASS,
        shuffle=False,
        preprocess_workers=PREPROCESS_WORKERS,
        verbose=verbose,
    )
    del data_test
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
    time_elapsed = time.time() - start_time
    if verbose:
        print(f"Data loaders created in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s")
    return train_loader, test_loader, int(time_elapsed)

def main() -> None:
    net = DenseNet(HIDDEN_SIZE)
    net.to(DEVICE)

    base_optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

    criterion = SignAwareMAE(
        sign_penalty_weight=0.0,
    )

    trainer = Trainer(
        net=net,
        get_data_loader_func=get_dataloaders,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        loss_plot_path=LOSS_PLOT_PATH,
        model_path=MODEL_PATH,
        middle_n_games=100,
        final_n_games=500,
        data_loader_update_per_epoch=DATA_LOADER_UPDATE_PER_EPOCH,
        verbose=True,
    )
    cnt = 0
    for _ in trainer.train(N_EPOCHS):
        cnt += 1
    print(f"Training completed in {cnt} epochs")

def export():
    net = DenseNet(HIDDEN_SIZE)
    net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    net.eval()
    base64_str = net.save_weights_base64()
    with open("weights.txt", "w") as f:
        f.write(base64_str)
