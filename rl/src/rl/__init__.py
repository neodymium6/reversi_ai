from rust_reversi import Board, Turn
import torch
import torchinfo
from rl.models.dense import DenseNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

def main():
    net: torch.nn.Module = DenseNet(128, 128, 64)
    net.to(DEVICE)
    torchinfo.summary(net, input_size=(BATCH_SIZE, 128), device=DEVICE)
    for param in net.parameters():
        print(f"Device: {param.device}")
        break
