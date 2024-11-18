from collections import namedtuple, deque
import random
import torch
import numpy as np

class ExperienceReplayBuffer(object):
    def __init__(self, buffer_size: int = 100000, batch_size: int = 64, seed: int = 42, device: str = "cuda"):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device
    
        self.memory = deque(maxlen=self.buffer_size)
        self.experiences = namedtuple(typename="experiences",
                                      field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, experiences):
        # experiences is a tuple of state, action, reward, next_state and done
        exp = self.experiences(*experiences)
        self.memory.append(exp)

    def sample(self):
        #sample a batch of experiences
        batch = random.sample(population=self.memory,
                              k=self.batch_size)
        # batch now is a list of namedtuples, unpack each element from it
        state = torch.from_numpy(np.stack([exp.state for exp in batch],axis=0)).to(dtype=torch.float32,device=self.device)
        action = torch.from_numpy(np.stack([exp.action for exp in batch],axis=0)).to(device=self.device, dtype=torch.int64)
        reward = torch.from_numpy(np.stack([exp.reward for exp in batch],axis=0)).to(dtype=torch.float32,device=self.device)
        next_state = torch.from_numpy(np.stack([exp.next_state for exp in batch],axis=0)).to(dtype=torch.float32, device=self.device)
        done = torch.from_numpy(np.stack([exp.done for exp in batch],axis=0)).to(device=self.device, dtype=torch.uint8)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)