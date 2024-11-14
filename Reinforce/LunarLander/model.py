from torch import nn
import torch
from torch.nn import functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, input_dims = 4, hidden_dim = 16, n_actions = 4, device = 'cuda', seed = 42):
        super().__init__()

        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, n_actions)
        self.v = nn.Linear(hidden_dim * 2, 1)

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state):
        # convert the state to a torch tensor
        state = torch.FloatTensor(state, device=self.device).unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = F.softmax(self.fc3(x), dim=1)
        return out

    def act(self, state):
        x = self(state).cpu()
        dist = Categorical(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
