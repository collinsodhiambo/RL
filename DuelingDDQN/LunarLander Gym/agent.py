from torch.optim import Adam
from utils import ReplayBuffer
from model import DuelingDDQN
from torch.nn import functional as F
import random
import torch

BUFFER_SIZE = 100000
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-6
LEARN_EVERY = 4
TAU = 1e-2 # for soft update
GAMMA = 0.99
GRAD_NORM_CLIP = 10.0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Agent:
    def __init__(self, state_dim = 8, hidden_dim = 64, action_dim = 4, gamma = GAMMA, seed = 42):
        
        self.action_dim = action_dim
        self.target_model = DuelingDDQN(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, seed = seed)
        self.local_model = DuelingDDQN(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, seed = seed)
        self.target_model.load_state_dict(self.local_model.state_dict())

        self.optimizer = Adam(params=self.local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.seed = seed
        self.gamma = gamma

        self.cur_step = 0

        self.memory = ReplayBuffer(maxlen=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=self.seed)

        torch.manual_seed(self.seed)

    def act(self, state, eps = 0.0):

        if random.random() < eps:
            action = random.choice(torch.arange(self.action_dim).data.numpy())

        else:
            self.local_model.eval()
            with torch.no_grad():
                action = self.local_model(torch.from_numpy(state).unsqueeze(0).to(DEVICE)).argmax().data.cpu().numpy()
            self.local_model.train()
        return action

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.cur_step = (self.cur_step + 1) % LEARN_EVERY

        if self.cur_step == 0:
            if len(self.memory) > BATCH_SIZE:
                self.learn()


    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        self.optimizer.zero_grad()
        current = self.local_model(states).gather(1, actions)

        next_actions_qs = self.local_model(next_states).max(1)[1].unsqueeze(1)
        target_ = self.target_model(next_states).detach().gather(1, next_actions_qs)
        target = rewards + self.gamma * (1 - dones) * target_

        loss = F.mse_loss(input=current, target=target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=GRAD_NORM_CLIP)
        self.optimizer.step()

        # soft update
        self.soft_update(self.local_model, self.target_model)
        
    def soft_update(self, local_model, target_model):
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
