import numpy as np
import h5py
from rust_reversi import Board, Turn
import torch
from distillation.models.transfomer import Transformer
from distillation.models.dense_v import DenseNetV
from distillation.models import ReversiNet
from sklearn.model_selection import train_test_split
import tqdm
from distillation.vs import vs_random, vs_mcts, vs_alpha_beta
import matplotlib.pyplot as plt
from distillation.dataset import DistillationDataset
from distillation.scheduler import TemperatureScheduler

MCTS_DATA_PATH = "data/mcts_boards.h5"
MCTS_DATA2_PATH = "data/mcts_boards2.h5"
WTHOR_DATA_PATH = "data/wthor_boards.h5"
TEACHER_MODEL_PATH = "models/teacher_model.pth"
STUDENT_MODEL_PATH = "models/student_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY =1e-5
N_EPOCHS = 10
MAX_DATA = int(1e4)
TEMPERATURE_START = 1.5
TEMPERATURE_END = 1.0
COOLING_PHASE_RATIO = 0.8
COMPOSITE_LOSS_ALPHA = 0.7
teacher_net: ReversiNet = Transformer(
    patch_size=2,
    embed_dim=160,
    num_heads=5,
    num_layers=8,
    mlp_ratio=4.0,
    dropout=0.0,
)
student_net: ReversiNet = DenseNetV(hidden_size=64)

def load_data() -> np.ndarray:
    with h5py.File(MCTS_DATA_PATH, "r") as f:
        mcts_data = f["data"][:]
        print(f"Loaded MCTS data with shape: {mcts_data.shape}")
    with h5py.File(MCTS_DATA2_PATH, "r") as f:
        mcts_data2 = f["data"][:]
        print(f"Loaded MCTS2 data with shape: {mcts_data2.shape}")
    with h5py.File(WTHOR_DATA_PATH, "r") as f:
        wthor_data = f["data"][:]
        print(f"Loaded WTHOR data with shape: {wthor_data.shape}")
    data: np.ndarray = np.concatenate([mcts_data, wthor_data], axis=0)
    data = np.concatenate([data, mcts_data2], axis=0)
    if data.shape[0] > MAX_DATA:
        print(f"Trimming data to {MAX_DATA}")
        data = np.random.choice(data, MAX_DATA, replace=False)
    return data


def train_model(data: np.ndarray) -> None:
    print("Training model...")
    # load teacher model
    state_dict = torch.load(TEACHER_MODEL_PATH, weights_only=True)
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace('_orig_mod.', '')
        new_state_dict[new_key] = state_dict[key]
    teacher_net.load_state_dict(new_state_dict)
    teacher_net.eval()
    teacher_net.to(DEVICE)
    student_net.to(DEVICE)

    # init data loaders
    X_train, X_test = train_test_split(data, test_size=0.1, shuffle=True)
    train_dataset = DistillationDataset(
        X_train,
        teacher_net,
        student_net,
        DEVICE,
        BATCH_SIZE,
    )
    test_dataset = DistillationDataset(
        X_test,
        teacher_net,
        student_net,
        DEVICE,
        BATCH_SIZE,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # init optimizer
    optimizer = torch.optim.AdamW(student_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # init criterion
    criterion = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=len(train_loader) * N_EPOCHS,
    )

    temperature_scheduler = TemperatureScheduler(
        TEMPERATURE_START,
        TEMPERATURE_END,
        N_EPOCHS * len(train_loader),
        COOLING_PHASE_RATIO,
    )

    train_losses = []
    test_losses = []
    test_temperatured_losses = []
    lrs = []
    # train loop
    epoch_pbar = tqdm.tqdm(range(N_EPOCHS), desc="Epoch", leave=False)
    for epoch in epoch_pbar:
        # print(f"Temperature: {temperature_scheduler.get_temperature():.4f}")
        student_net.train()
        pb = tqdm.tqdm(total=len(train_loader), leave=False)
        train_loss = 0.0
        for i, (student_input, teacher_v) in enumerate(train_loader):
            student_input: torch.Tensor = student_input.to(DEVICE)
            optimizer.zero_grad()
            student_v = student_net(student_input)
            hard_loss: torch.Tensor = criterion(student_v, teacher_v)
            teacher_v = temperature_scheduler.temp_teacher(teacher_v)
            soft_loss: torch.Tensor = criterion(student_v, teacher_v)
            loss = COMPOSITE_LOSS_ALPHA * soft_loss + (1.0 - COMPOSITE_LOSS_ALPHA) * hard_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            temperature_scheduler.step()
            pb.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
            train_loss += loss.item()
            pb.update(1)
        pb.close()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        student_net.eval()
        torch.save(student_net.state_dict(), STUDENT_MODEL_PATH)
        with torch.no_grad():
            test_loss = 0.0
            test_tempatured_loss = 0.0
            pb = tqdm.tqdm(total=len(test_loader), leave=False)
            for student_input, teacher_v in test_loader:
                student_input = student_input.to(DEVICE)
                student_v = student_net(student_input)
                loss = criterion(student_v, teacher_v)
                test_loss += loss.item()

                teacher_v = temperature_scheduler.temp_teacher(teacher_v)
                loss = criterion(student_v, teacher_v)
                test_tempatured_loss += loss.item()
                pb.set_description(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Tempatured Loss: {test_tempatured_loss:.4f}")
                pb.update(1)
            pb.close()

            test_loss /= len(test_loader)
            test_tempatured_loss /= len(test_loader)
            n_games = 100
            random_win_rate = vs_random(n_games, student_net)
            mcts_win_rate = vs_mcts(n_games, student_net)
            alpha_beta_win_rate = vs_alpha_beta(n_games, student_net)
            epoch_pbar.write(
                f"Epoch {epoch:{len(str(N_EPOCHS))}d}: " 
                + f"Test Loss: {test_loss:.4f}, Test Tempatured Loss: {test_tempatured_loss:.4f}, "
                + f"Random Win Rate: {random_win_rate: .4f}, MCTS Win Rate: {mcts_win_rate: .4f}, AlphaBeta Win Rate: {alpha_beta_win_rate:.4f}"
            )

        test_losses.append(test_loss)
        test_temperatured_losses.append(test_tempatured_loss)
        lrs.append(scheduler.get_last_lr()[0])
        # plot losses
        fig, ax = plt.subplots()
        ax.plot(train_losses, label="Train Loss")
        ax.plot(test_losses, label="Test Loss")
        ax.plot(test_temperatured_losses, label="Test Tempatured Loss")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid()
        ax.legend(loc="upper left")
        ax2 = ax.twinx()
        ax2.plot(lrs, color="red", label="Learning Rate", linestyle="--")
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale("log")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("loss.png", dpi=300)
        plt.close()

    n_games = 300
    random_win_rate = vs_random(n_games, student_net)
    mcts_win_rate = vs_mcts(n_games, student_net)
    alpha_beta_win_rate = vs_alpha_beta(n_games, student_net)
    print(
        f"Epoch {epoch:{len(str(N_EPOCHS))}d}: " 
        + f"Test Loss: {test_loss:.4f}, Test Tempatured Loss: {test_tempatured_loss:.4f}, "
        + f"Random Win Rate: {random_win_rate: .4f}, MCTS Win Rate: {mcts_win_rate: .4f}, AlphaBeta Win Rate: {alpha_beta_win_rate:.4f}"
    )


def main() -> None:
    data = load_data()
    try:
        train_model(data)
    except KeyboardInterrupt:
        print("Training interrupted")
        return


def export_student_model() -> None:
    student_net.load_state_dict(torch.load(STUDENT_MODEL_PATH))
    student_net.eval()
    base64_str = student_net.save_weights_base64()
    with open("dense.txt", "w") as f:
        f.write(base64_str)
