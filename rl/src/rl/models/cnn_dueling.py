import torch

DROPOUT = 0.0
# 8x8 board + 1 for pass
OUTPUT_SIZE = 65

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvLayer, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(DROPOUT)
    
    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.relu(x)
        x = self.dropout(x)
        return x

class Conv5DuelingNet(torch.nn.Module):
    def __init__(self, num_channels, fc_hidden_size):
        super(Conv5DuelingNet, self).__init__()
        self.conv1 = ConvLayer(2, num_channels, kernel_size=3, padding=1)
        self.conv2 = ConvLayer(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = ConvLayer(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv4 = ConvLayer(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv5 = ConvLayer(num_channels, num_channels, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(num_channels * 8 * 8, fc_hidden_size)
        self.fc2_adv = torch.nn.Linear(fc_hidden_size, OUTPUT_SIZE)
        self.fc2_val = torch.nn.Linear(fc_hidden_size, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, self.num_channels * 8 * 8)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        adv = self.fc2_adv(x)
        val = self.fc2_val(x).expand(-1, OUTPUT_SIZE)

        return val + adv - adv.mean(1, keepdim=True).expand(-1, OUTPUT_SIZE)
