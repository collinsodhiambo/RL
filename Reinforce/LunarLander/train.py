import gymnasium as gym
import torch
import numpy as np
from model import Policy as PG
from torch.optim import Adam
from collections import deque
import matplotlib.pyplot as plt

seed = 42
torch.manual_seed(seed)

name = 'LunarLander-v2'
env = gym.make(name)
threshold = gym.envs.registry.get(name).reward_threshold
env.reset(seed=seed)

n_actions = env.action_space.n
input_dim = env.observation_space.shape[0]
hidden_dim = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-3
gamma = 0.99

net = PG(input_dims=input_dim, hidden_dim=hidden_dim,
         n_actions=n_actions,device=device, seed=seed)

optimizer = Adam(net.parameters(),lr=lr)

def train(episodes = 5000, max_iter = 1000):
    scores = []
    scores_window = deque(maxlen=100)


    for episode_idx in range(1, episodes + 1):
        log_probs = []
        rewards = []
        score = 0.0

        state, _ = env.reset()
        for i in range(max_iter):
            action, log_prob = net.act(state)
            state, reward, done, trunc, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            score += reward
            if done or trunc:
                break
        
        scores.append(score)
        scores_window.append(score)

        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            for i, r in enumerate(rewards[t:]):
                Gt += gamma ** i * r
            discounted_rewards.append(Gt)
            
        discounted_rewards =  torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean())/ (discounted_rewards.std() + 1e-9)
        
        grad_log_probs_list = []
        for log_prob, R in zip(log_probs, discounted_rewards):
            grad_log_probs_list.append(-log_prob * R)

        grads = torch.cat(grad_log_probs_list).sum()
        optimizer.zero_grad()
        grads.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        optimizer.step()

        if episode_idx % 100 == 0:
            print(f"Episode: {episode_idx}\tAverage Score: {np.mean(scores_window)}")
        
        if np.mean(scores_window) >= threshold + 29:
            print(f"Episode: {episode_idx}\tAverage Score: {np.mean(scores_window)}")
            print(f"Environment solved in {episode_idx} steps! Mean score: {np.mean(scores_window)}")

            data = {
                'model_state_dict':net.state_dict()
            }
            torch.save(data, f"model.pt")
            break

    return scores

scores = train()

plt.plot(np.arange(len(scores)), scores)
plt.xlabel("# Episodes")
plt.ylabel("Scores")
plt.savefig("Lunar_lander_reinforce_results.png")
plt.show()