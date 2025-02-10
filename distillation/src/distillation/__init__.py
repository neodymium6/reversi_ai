import json
import numpy as np
import h5py
import torch
from distillation.models.transfomer import Transformer
from distillation.models.dense import DenseNet
from distillation.models import ReversiNet
from sklearn.model_selection import train_test_split
import tqdm
from distillation.vs import vs_random, vs_mcts, vs_alpha_beta
from distillation.losses.mse_ranking import MSEWithRankingLoss

MCTS_DATA_PATH = "data/mcts_boards.h5"
MCTS_DATA2_PATH = "data/mcts_boards2.h5"
WTHOR_DATA_PATH = "data/wthor_boards.h5"
TEACHER_MODEL_PATH = "models/teacher_model.pth"
STUDENT_MODEL_PATH = "models/student_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
LR = 1e-4
WEIGHT_DECAY =1e-7
N_EPOCHS = 10
MAX_DATA = int(2e5)
teacher_net: ReversiNet = Transformer(
    patch_size=2,
    embed_dim=160,
    num_heads=5,
    num_layers=8,
    mlp_ratio=4.0,
    dropout=0.0,
)
student_net: ReversiNet = DenseNet(hidden_size=128)

class DistillationDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self):
        return len(self.X)
    
    def _x2teacher_input(self, x: np.void) -> torch.Tensor:
        return teacher_net.x2input(x)
    
    def _x2student_input(self, x: np.void) -> torch.Tensor:
        return student_net.x2input(x)

    def __getitem__(self, idx):
        return (self._x2teacher_input(self.X[idx]), self._x2student_input(self.X[idx]))

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
    train_dataset = DistillationDataset(X_train)
    test_dataset = DistillationDataset(X_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # init optimizer
    optimizer = torch.optim.AdamW(student_net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=N_EPOCHS * len(train_loader),
        final_div_factor=1e5,
    )

    # init criterion
    criterion = MSEWithRankingLoss(rank_weight=2.0)

    # train loop
    for epoch in range(N_EPOCHS):
        student_net.train()
        pb = tqdm.tqdm(total=len(train_loader))
        for i, (teacher_input, student_input) in enumerate(train_loader):
            teacher_input: torch.Tensor = teacher_input.to(DEVICE)
            student_input: torch.Tensor = student_input.to(DEVICE)
            optimizer.zero_grad()
            teacher_output = teacher_net(teacher_input)
            student_output = student_net(student_input)
            loss: torch.Tensor = criterion(student_output, teacher_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pb.update(1)
        pb.close()
        student_net.eval()
        with torch.no_grad():
            test_loss = 0.0
            for i, (teacher_input, student_input) in enumerate(test_loader):
                teacher_input = teacher_input.to(DEVICE)
                student_input = student_input.to(DEVICE)
                teacher_output = teacher_net(teacher_input)
                student_output = student_net(student_input)
                loss = criterion(student_output, teacher_output)
                test_loss += loss.item()
            test_loss /= len(test_loader)
            print(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}")
            random_win_rate = vs_random(100, student_net)
            mcts_win_rate = vs_mcts(100, student_net)
            alpha_beta_win_rate = vs_alpha_beta(100, student_net)
            print(f"Random Win Rate: {random_win_rate}, MCTS Win Rate: {mcts_win_rate}, AlphaBeta Win Rate: {alpha_beta_win_rate}")
        torch.save(student_net.state_dict(), STUDENT_MODEL_PATH)

    random_win_rate = vs_random(1000, student_net)
    mcts_win_rate = vs_mcts(1000, student_net)
    alpha_beta_win_rate = vs_alpha_beta(1000, student_net)
    print(f"Final Random Win Rate: {random_win_rate}, MCTS Win Rate: {mcts_win_rate}, AlphaBeta Win Rate: {alpha_beta_win_rate}")


def main() -> None:
    data = load_data()
    train_model(data)

def export_student_model() -> None:
    student_net = DenseNet(hidden_size=128)
    student_net.load_state_dict(torch.load(STUDENT_MODEL_PATH))

    params = {}

    params['ih_weights'] = student_net.fc1.weight.detach().numpy().tolist()
    params['h1_biases'] = student_net.fc1.bias.detach().numpy().tolist()
    params['ho_weights'] = student_net.fc2.weight.detach().numpy().tolist()
    params['o_biases'] = student_net.fc2.bias.detach().numpy().tolist()

    with open("dense.json", "w") as f:
        json.dump(params, f)
