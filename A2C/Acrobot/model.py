import torch
from torch import nn
from torch.nn import functional as F

class A2C(nn.Module):
    def __init__(self, in_features = 8, out_features = 16, num_actions = 4, device = 'cpu', seed = 42):
        super().__init__()

        torch.manual_seed(seed)
        self.device = device
        self.backbone = nn.Sequential(
            nn.Linear(in_features, out_features, device=self.device),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

        self.policy_head = nn.Linear(out_features, num_actions)
        self.value_head = nn.Linear(out_features, 1)

    def forward(self, state):
        x = torch.FloatTensor(state, device=self.device).unsqueeze(0)
        common = F.relu(self.backbone(x))
        v = self.value_head(common)
        policy_dist = F.softmax(self.policy_head(common), dim=1)
        return policy_dist, v