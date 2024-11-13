import torch
import gymnasium as gym
from agent import Agent
from time import sleep

env = gym.make("CartPole-v1", render_mode = "human")
net = Agent(state_dim=env.observation_space.sample().shape[0], hidden_dim=64, action_dim=env.action_space.n)

chkpt = torch.load("/home/collins/RL/DuelingDDQN/CartPole Gym/net_checkpoint_episodes_443_2.pt",
                   map_location = "cpu", weights_only = True)
net.local_model.load_state_dict(chkpt["model_state_dict"])

def play(n = 2):
    for i in range(n):
        state = env.reset()
        total_reward = 0
        while True:
            env.render()
            action = net.act(state[0] if isinstance(state, tuple) else state)
            state, reward, done, trunc, info = env.step(action)
            total_reward += reward
            if done:
                break
        
        print(f"{i+1}/{n}: Total reward: {total_reward}")         
    env.close()
play(n=2)
