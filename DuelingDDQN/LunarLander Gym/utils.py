from collections import deque, namedtuple
import random
from torch import float32, tensor, uint8, int64
from numpy import vstack
import torch

class ReplayBuffer:
    def __init__(self, maxlen = 50000,batch_size = 64, seed = 42):

        self.batch_size = batch_size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=maxlen)
        self.experience = namedtuple("experiences", field_names="state action reward next_state done".split(sep=" "))
        
    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):

        batch = random.sample(self.memory, self.batch_size)

        state = tensor(vstack([ex.state for ex in batch if ex is not None]), dtype=float32, device=self.device)
        action = tensor(vstack([ex.action for ex in batch if ex is not None]), dtype=int64, device=self.device)
        reward = tensor(vstack([ex.reward for ex in batch if ex is not None]), dtype=float32, device=self.device)
        next_state = tensor(vstack([ex.next_state for ex in batch if ex is not None]), dtype=float32, device=self.device)
        done = tensor(vstack([ex.done for ex in batch if ex is not None]), dtype=uint8, device=self.device)

        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.memory)

