import torch

class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        identity = x
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        x += identity
        x = self.relu(x)
        return x

class ResNet10(torch.nn.Module):
    def __init__(self, num_channels, fc_hidden_size):
        super(ResNet10, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.res_blocks = torch.nn.ModuleList([ResBlock(num_channels) for _ in range(10)])
        self.fc1 = torch.nn.Linear(num_channels * 8 * 8, fc_hidden_size)
        self.fc2_adv = torch.nn.Linear(fc_hidden_size, 64)
        self.fc2_val = torch.nn.Linear(fc_hidden_size, 1)
        self.num_channels = num_channels
        self.relu = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.bn1(self.conv1(x))
        x = self.relu(x)

        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = x.view(-1, self.num_channels * 8 * 8)
        x = self.fc1(x)
        x = self.relu(x)

        adv = self.fc2_adv(x)
        val = self.fc2_val(x).expand(-1, 64)

        return val + adv - adv.mean(1, keepdim=True).expand(-1, 64)
