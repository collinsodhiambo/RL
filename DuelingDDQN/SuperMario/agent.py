from utils import ExperienceReplayBuffer
from model import DuelingDDQN as DDQN
import torch
from torch.optim import Adam
from torch.nn import functional as F
import random
import numpy as np
import torchvision.transforms as T

class SuperMarioAgent:
    def __init__(self, input_dim: tuple = (4, 84, 84), action_dim: int = 4,
                 hidden_dim: int = 32, lr: float = 5e-4, buffer_size: int = 100000,
                 batch_size: int = 64, TAU: float = 1e-2, GAMMA: float = 0.99,
                 WEIGHT_NORM_CLIP: float = 10.0, LEARN_EVERY = 4,
                 seed: int = 42, device: str = "cuda"):
        
        c, h, w = input_dim
        torch.manual_seed(seed)
        self.action_dim = action_dim
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.device = device
        self.memory = ExperienceReplayBuffer(buffer_size=buffer_size,
                                             batch_size=batch_size, seed=seed, device=self.device)
        self.local_model = DDQN(in_channels=c, hidden_dim=hidden_dim,action_dim=action_dim,
                                seed=seed)
        self.target_model = DDQN(in_channels=c, hidden_dim=hidden_dim,action_dim=action_dim,
                                seed=seed)
        self.target_model.load_state_dict(self.local_model.state_dict())
        self.optimizer = Adam(self.local_model.parameters(),
                              lr = lr)
        
        self.GAMMA = GAMMA
        self.LEARN_EVERY = LEARN_EVERY
        self.cur_step = 0
        self.TAU = TAU
        self.clip_ = WEIGHT_NORM_CLIP

        self.target_model.to(device=self.device)
        self.local_model.to(device=self.device)

        self.transforms = T.Compose(
            [T.Normalize(0, 255)]
        )

    def act(self, state, eps = 0.0):

        #epsilon greedy strategy, decreasing epsilon
        if random.random() > eps:
            self.local_model.eval()
            with torch.no_grad():
                state = self.transforms(torch.from_numpy(state).to(dtype=torch.float32, device = self.device)).unsqueeze(0)
                action_values = self.local_model(state)
                action = action_values.argmax().detach().cpu().item()
            self.local_model.train()
        else:
            action = random.choice(np.arange(self.action_dim))

        return action

    def step(self, state, action, reward, next_state, done):
        experiences = (state, action, reward, next_state, done)
        self.memory.add(experiences=experiences)

        self.cur_step += 1
        if self.cur_step % self.LEARN_EVERY == 0:
            if len(self.memory) > self.batch_size:
                self.learn()

    def learn(self):
        state, action, reward, next_state, done = self.memory.sample()
        state = self.transforms(state)
        next_state = self.transforms(next_state)


        self.optimizer.zero_grad()
        current = self.local_model(state)[torch.arange(self.batch_size), action]

        next_action = self.local_model(next_state).argmax(1)[1]
        target_ = self.target_model(next_state).detach()[torch.arange(self.batch_size), next_action]
        target = reward + self.GAMMA * (1 - done) * target_

        loss = F.mse_loss(current, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=self.clip_)
        self.optimizer.step()

        self.soft_update()

    def soft_update(self):
        
        for local_param, target_param in zip(self.local_model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1 - self.TAU) * target_param.data)
    
