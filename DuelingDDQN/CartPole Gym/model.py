import torch
from torch import nn
# from torch.nn import functional as F

class DuelingDDQN(nn.Module):
    def __init__(self, state_dim = 8, hidden_dim = 64, action_dim = 4, seed = 42):
        super(DuelingDDQN, self).__init__()
        torch.manual_seed(seed)
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU()
        )
        self.adv = nn.Linear(hidden_dim * 4, action_dim)
        self.v = nn.Linear(hidden_dim * 4, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        x = self.body(state)
        adv, v = self.adv(x), self.v(x)
        return v + (adv -  adv.mean(dim = 1, keepdim = True)) #adv.mean(dim = 1, keepdim = True)
