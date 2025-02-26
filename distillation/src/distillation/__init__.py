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

MCTS_DATA_PATH = "data/mcts_boards.h5"
MCTS_DATA2_PATH = "data/mcts_boards2.h5"
WTHOR_DATA_PATH = "data/wthor_boards.h5"
TEACHER_MODEL_PATH = "models/teacher_model.pth"
STUDENT_MODEL_PATH = "models/student_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4096
LR = 1e-4
WEIGHT_DECAY =1e-5
N_EPOCHS = 10
MAX_DATA = int(2e6)
TEMPRATURE_START = 1.5
TEMPRATURE_END = 1.0
teacher_net: ReversiNet = Transformer(
    patch_size=2,
    embed_dim=160,
    num_heads=5,
    num_layers=8,
    mlp_ratio=4.0,
    dropout=0.0,
)
student_net: ReversiNet = DenseNetV(hidden_size=128)

class DistillationDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X
        self.teacher_v = torch.zeros(len(X), 1, dtype=torch.float32)
        idx = 0
        pb = tqdm.tqdm(total=len(X) // BATCH_SIZE)
        pb.set_description("Initializing Distillation Dataset")
        while idx + BATCH_SIZE < len(X):
            teacher_input = torch.stack([self._x2teacher_input(x) for x in X[idx:idx+BATCH_SIZE]], dim=0).to(DEVICE)
            legal_actions = torch.stack([self._legal_actions(x) for x in X[idx:idx+BATCH_SIZE]], dim=0)
            with torch.no_grad():
                teacher_output = teacher_net(teacher_input)
                teacher_output = teacher_output.masked_fill_(~legal_actions, -1e9)
                teacher_v = torch.max(teacher_output, dim=1, keepdim=True).values
                self.teacher_v[idx:idx+BATCH_SIZE] = teacher_v
            idx += BATCH_SIZE
            pb.update(1)
        pb.close()
        teacher_input = torch.stack([self._x2teacher_input(x) for x in X[idx:]], dim=0).to(DEVICE)
        legal_actions = torch.stack([self._legal_actions(x) for x in X[idx:]], dim=0)
        with torch.no_grad():
            teacher_output = teacher_net(teacher_input)
            teacher_output = teacher_output.masked_fill_(~legal_actions, -1e9)
            teacher_v = torch.max(teacher_output, dim=1, keepdim=True).values
            self.teacher_v[idx:] = teacher_v
        self.teacher_v = self.teacher_v.to(DEVICE)

    def __len__(self):
        return len(self.X)
    
    def _x2teacher_input(self, x: np.void) -> torch.Tensor:
        return teacher_net.x2input(x)
    
    def _x2student_input(self, x: np.void) -> torch.Tensor:
        return student_net.x2input(x)

    def _legal_actions(self, x: np.void) -> torch.Tensor:
        player_board = x["player_board"]
        opponent_board = x["opponent_board"]
        turn_str = x["turn"]
        if turn_str == b'Black':
            turn = Turn.BLACK
        else:
            turn = Turn.WHITE
        board = Board()
        board.set_board(player_board, opponent_board, turn)
        legal_actions = torch.tensor(
            board.get_legal_moves_tf() + [board.is_pass()],
            dtype=torch.bool,
            device=DEVICE,
        )
        return legal_actions

    def __getitem__(self, idx):
        return (self._x2student_input(self.X[idx]), self.teacher_v[idx])

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

def temp_teacher(teacher_v: torch.Tensor, temperature: float) -> torch.Tensor:
    return 0.5 + (teacher_v - 0.5) / temperature

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
    train_dataset = DistillationDataset(X_train)
    test_dataset = DistillationDataset(X_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # init optimizer
    optimizer = torch.optim.AdamW(student_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # init criterion
    criterion = torch.nn.MSELoss()

    train_losses = []
    test_losses = []
    test_tempatured_losses = []
    # train loop
    for epoch in range(N_EPOCHS):
        temprature = TEMPRATURE_START + (TEMPRATURE_START - TEMPRATURE_END) * epoch / (1 - N_EPOCHS)
        print(f"Temperature: {temprature:.4f}")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LR,
            total_steps=len(train_loader),
        )
        student_net.train()
        pb = tqdm.tqdm(total=len(train_loader))
        for i, (student_input, teacher_v) in enumerate(train_loader):
            student_input: torch.Tensor = student_input.to(DEVICE)
            optimizer.zero_grad()
            teacher_v = temp_teacher(teacher_v, temprature)

            student_v = student_net(student_input)
            loss: torch.Tensor = criterion(student_v, teacher_v)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pb.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
            train_losses.append((epoch + i / len(train_loader), loss.item()))
            pb.update(1)
        pb.close()
        student_net.eval()
        torch.save(student_net.state_dict(), STUDENT_MODEL_PATH)
        with torch.no_grad():
            test_loss = 0.0
            test_tempatured_loss = 0.0
            for student_input, teacher_v in test_loader:
                student_input = student_input.to(DEVICE)
                student_v = student_net(student_input)
                loss = criterion(student_v, teacher_v)
                test_loss += loss.item()

                teacher_v = temp_teacher(teacher_v, temprature)
                loss = criterion(student_v, teacher_v)
                test_tempatured_loss += loss.item()

            test_loss /= len(test_loader)
            test_tempatured_loss /= len(test_loader)
            print(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Tempatured Loss: {test_tempatured_loss:.4f}")
            n_games = 30
            random_win_rate = vs_random(n_games, student_net)
            mcts_win_rate = vs_mcts(n_games, student_net)
            alpha_beta_win_rate = vs_alpha_beta(n_games, student_net)
            print(f"Random Win Rate: {random_win_rate: .4f}, MCTS Win Rate: {mcts_win_rate: .4f}, AlphaBeta Win Rate: {alpha_beta_win_rate: .4f}")

        test_losses.append((epoch+1, test_loss))
        test_tempatured_losses.append((epoch+1, test_tempatured_loss))
        # plot losses
        fig, ax = plt.subplots()
        ax.plot([x[0] for x in train_losses], [x[1] for x in train_losses], label="Train Loss")
        ax.plot([x[0] for x in test_losses], [x[1] for x in test_losses], label="Test Loss")
        ax.plot([x[0] for x in test_tempatured_losses], [x[1] for x in test_tempatured_losses], label="Test Tempatured Loss")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid()
        ax.legend()
        plt.tight_layout()
        plt.savefig("loss.png", dpi=300)
        plt.close()

    n_games = 100
    random_win_rate = vs_random(n_games, student_net)
    mcts_win_rate = vs_mcts(n_games, student_net)
    alpha_beta_win_rate = vs_alpha_beta(n_games, student_net)
    print(f"Final Random Win Rate: {random_win_rate: .4f}, MCTS Win Rate: {mcts_win_rate: .4f}, AlphaBeta Win Rate: {alpha_beta_win_rate: .4f}")


def main() -> None:
    data = load_data()
    train_model(data)

def export_student_model() -> None:
    student_net.load_state_dict(torch.load(STUDENT_MODEL_PATH))
    student_net.eval()
    base64_str = student_net.save_weights_base64()
    with open("dense.txt", "w") as f:
        f.write(base64_str)
