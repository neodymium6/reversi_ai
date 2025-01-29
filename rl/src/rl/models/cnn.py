import torch

class Conv5Net(torch.nn.Module):
    def __init__(self, num_channels, fc_hidden_size):
        super(Conv5Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, num_channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)
        self.conv3 = torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(num_channels)
        self.conv4 = torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(num_channels)
        self.conv5 = torch.nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(num_channels)
        self.fc1 = torch.nn.Linear(num_channels * 8 * 8, fc_hidden_size)
        self.fc2 = torch.nn.Linear(fc_hidden_size, 64)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn3(self.conv3(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn4(self.conv4(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn5(self.conv5(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = x.view(-1, self.num_channels * 8 * 8)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
