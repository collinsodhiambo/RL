import torch
from model import A2C
import gymnasium as gym
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('model.pt', weights_only=True,
                  map_location=device)

env = gym.make('Acrobot-v1', render_mode = 'human')
env.reset(seed=35)
np.random.seed(35)


n_actions = env.action_space.n
input_dim = env.observation_space.shape[0]
hidden_dim = 256

net = A2C(input_dim, hidden_dim,
         n_actions,device=device, seed=42)
net.load_state_dict(ckpt)

def play(n=5):
    for i in range(n):
        state, _ = env.reset()
        total = 0.0
        while True:
            logits, _ = net(state)
            action = np.random.choice(n_actions, p=np.squeeze(logits.detach().numpy()))
            state, reward, done, trunc, _ = env.step(action)
            total += reward

            if done or trunc:
                break

        print(total)
    env.close()

play(2)
