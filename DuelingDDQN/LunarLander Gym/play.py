import torch
import gymnasium as gym
from agent import Agent

env = gym.make("LunarLander-v2", render_mode = 'human')

net = Agent(state_dim=8, hidden_dim=64, action_dim=env.action_space.n)
# state = env.reset(seed=42)

chkpt = torch.load("net_checkpoint_episodes_267.pt",
                   map_location = "cpu")
net.local_model.load_state_dict(chkpt["model_state_dict"])

def play(n = 2):
    for i in range(n):
        state = env.reset()
        score = 0
        while True:
            action = net.act(state[0] if isinstance(state, tuple) else state)
            state, reward, done, trunc, info = env.step(action)
            score += reward
            if done:
                break
            env.render()

        print(score)
    env.close()
play(n=5)
