import gymnasium as gym
import torch
from torch.optim import Adam
from model import A2C
from collections import deque
import numpy as np
from matplotlib import pyplot as plt

seed = 42
name = 'CartPole-v0'
env = gym.make(name)
state, _ = env.reset(seed=seed)
reward_threshold = gym.envs.registry.get(name, 195.0).reward_threshold
max_iter = gym.envs.registry.get(name, 500).max_episode_steps
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dim = env.observation_space.shape
hidden_dim = 64
num_actions = env.action_space.n

lr = 2e-3
gamma = 0.99
BETA = 1e-2 # for exploration
CLIP = 10.0 # for gradient clip
net = A2C(in_features=input_dim, out_features=hidden_dim, num_actions=num_actions,
          device=device, seed = seed)
optimizer = Adam(net.parameters(), lr=lr, eps=1e-8)

def train(max_episodes = 5000, max_iter = 500):
    reward_window = deque(maxlen=100)
    scores = []

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        values = []
        score = 0.0
        entropy_term = 0.0

        for i in range(max_iter):
            logits, value = net(state)
            value = value.detach().numpy()[0,0]
            dist = logits.detach().numpy()

            action = np.random.choice(dist, p=np.squeeze(dist))
            log_prob = torch.log(logits.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            state, reward, done, trunc, _ = env.step(action)
            score += reward
            values.append(value)
            rewards.append(reward)
            log_probs.append(log_prob)
            entropy_term += entropy

            if done or trunc:
                R = 0
                break
        scores.append(score)
        reward_window.append(score)

        disc_rewards = np.zeros_like(rewards)

        for t in reversed(range(len(rewards))):
            R = rewards[t] + gamma * R
            disc_rewards[t] = R

        values = torch.FloatTensor(values)
        disc_rewards = torch.FloatTensor(disc_rewards)
        log_probs = torch.cat(log_probs)

        advantage = disc_rewards - values
        actor_loss = -(advantage * log_probs).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        loss = actor_loss + critic_loss + BETA * entropy_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode: {episode}\tAverage Reward: {np.mean(reward_window)}\t"
                  f"Critic Loss: {critic_loss.item()}\tActor Loss: {actor_loss.item()}\t"
                  f"Total Loss: {loss.item()}\tEntropy: {entropy_term}")
        if np.mean(reward_window) >= reward_threshold:
            print(f"Episode: {episode}\tAverage Reward: {np.mean(reward_window)}\t"
                  f"Critic Loss: {critic_loss.item()}\tActor Loss: {actor_loss.item()}\t"
                  f"Total Loss: {loss.item()}\tEntropy: {entropy_term}")
            print(f"Environment solved in {episode} steps! Mean reward: {np.mean(reward_window)}")
            torch.save(net.state_dict(), 'model.pt')
            break

    return scores

scores = train()
running_mean = [scores[0]]
p = 0.9
for score in scores:
    running_mean.append(p * running_mean[-1] + (1-p) * score)
running_mean = running_mean[1:]

plt.plot(np.arange(len(scores)), scores, label = 'score')
plt.plt(np.arange(len(scores)), running_mean, label = 'running_mean')
plt.xlabel("# Episodes")
plt.ylabel("# Score")
plt.savefig('scores.png')
plt.show()







