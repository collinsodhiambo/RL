from torch import nn
import torch
from torch.nn import functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, input_dims = 4, hidden_dim = 16, n_actions = 4, device = 'cuda', seed = 42):
        super().__init__()

        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state):
        # convert the state to a torch tensor
        state = torch.FloatTensor(state, device=self.device).unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x),dim=1)
        return x

    def act(self, state):
        x = self(state).cpu()
        dist = Categorical(x)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
