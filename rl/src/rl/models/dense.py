import torch

INPUT_SIZE = 128
# 8x8 board + 1 for pass
OUTPUT_SIZE = 65

class DenseNet(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super(DenseNet, self).__init__()
        self.fc1 = torch.nn.Linear(INPUT_SIZE, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, OUTPUT_SIZE)
        self.relu = torch.nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
