import torch
from torch import nn
from torch.nn import functional as F

class DuelingDDQN(nn.Module):
    def __init__(self, in_channels:int = 3, hidden_dim: int = 32, action_dim: int = 8, seed = 42):
        super().__init__()

        torch.manual_seed(seed=seed)

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,
                               kernel_size=(3,3),stride=2,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=hidden_dim) # 42

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2,
                               kernel_size=(4,4), stride=2, padding=1,bias=False) # 20
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)

        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4,
                               kernel_size=(3,3), stride=2, padding=1,bias=False) # 10

        self.resblock = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim // 2, (1,1), 1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, (3,3),1,bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim * 4, 1, 1,1, bias=False) # 10
        )
        self.bnres1 = nn.BatchNorm2d(hidden_dim * 4)

        self.conv4 = nn.Conv2d(hidden_dim * 4, hidden_dim * 4,
                               kernel_size=(3,3), stride=2, bias=False) # 5

        self.resblock2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim // 2, (1,1), 1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, (3,3),1,bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim * 4, 1, 1, 1, bias=False) # 5
        )
        self.bnres2 = nn.BatchNorm2d(hidden_dim * 4)

        out_shape = 5 * 5 * hidden_dim * 4
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=out_shape, out_features= out_shape // 25, bias=False)
        self.bnfc = nn.BatchNorm1d(out_shape // 25)

        self.fc2 = nn.Linear(in_features=out_shape // 25, out_features= out_shape // 25,bias=False)
        self.bnfc2 = nn.BatchNorm1d(out_shape // 25)

        self.adv = nn.Linear(out_shape // 25, action_dim)
        self.v = nn.Linear(out_shape // 25, 1)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x))) # 42
        x = F.relu(self.bn2(self.conv2(x))) # 20
        x = F.relu(self.conv3(x)) # 10

        x =  F.relu(self.bnres1(x + self.resblock(x))) # 10
        x = F.relu(self.conv4(x)) # 5

        x =  F.relu(self.bnres2(x + F.relu(self.resblock2(x)))) # 5

        x = self.flatten(x)

        x = F.relu(self.bnfc(self.fc(x)))
        x = F.relu(self.bnfc2(self.fc2(x)))

        adv = self.adv(x)
        v = self.v(x)

        return v + (adv - adv.mean(1, keepdim = True))
    

if __name__ == "__main__":
    net = DuelingDDQN(in_channels=1, hidden_dim=64)

    x = torch.randn(4, 1, 84, 84)
    out = net(x)
    print(out)
    print(sum([p.numel() for p in net.parameters()])/1e6)






        