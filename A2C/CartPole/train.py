import gymnasium as gym
import torch
from torch.optim import Adam
from model import A2C
from collections import deque
import numpy as np
from matplotlib import pyplot as plt

seed = 42
torch.manual_seed(seed)
np.random.seed(seed=seed)
name = 'CartPole-v1'
env = gym.make(name)
state, _ = env.reset(seed=seed)
reward_threshold = env.spec.reward_threshold
max_iter = env.spec.max_episode_steps
device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_dim = env.observation_space.shape[0]
hidden_dim = 128
num_actions = env.action_space.n

lr = 5e-3
gamma = 0.99
BETA = 1e-2 # for exploration
CLIP = 0.1 # for gradient clip
net = A2C(in_features=input_dim, out_features=hidden_dim, num_actions=num_actions,
          device=device, seed = seed)
optimizer = Adam(net.parameters(), lr=lr, eps=1e-8)

def train(max_episodes = 5000, max_iter = 500):
    print('[INFO]: Training')
    reward_window = deque(maxlen=100)
    scores = []
    entropy_term = 0.0

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        values = []
        score = 0.0

        for i in range(max_iter):
            logits, value = net(state)
            value = value.detach().numpy()[0,0]
            dist = logits.detach().numpy()

            action = np.random.choice(num_actions, p=np.squeeze(dist))
            log_prob = torch.log(logits.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            state, reward, done, trunc, _ = env.step(action)
            score += reward
            values.append(value)
            rewards.append(reward)
            log_probs.append(log_prob)
            entropy_term += entropy

            if done or trunc or i == max_iter:
                R = net(state)[1]
                R = R.detach().numpy()[0,0]
                break
        scores.append(score)
        reward_window.append(score)

        disc_rewards = np.zeros_like(rewards)

        for t in reversed(range(len(rewards))):
            R = rewards[t] + gamma * R
            disc_rewards[t] = R

        values = torch.FloatTensor(values)
        disc_rewards = torch.FloatTensor(disc_rewards)
        disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + 1e-8)
        log_probs = torch.stack(log_probs)

        advantage = disc_rewards - values
        actor_loss = -(advantage * log_probs).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        loss = actor_loss + critic_loss + BETA * entropy_term

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode: {episode}\tAverage Reward: {np.mean(reward_window)}")
        if np.mean(reward_window) >= reward_threshold:
            print(f"Episode: {episode}\tAverage Reward: {np.mean(reward_window)}")
            print(f"Environment solved in {episode} steps! Mean reward: {np.mean(reward_window)}")
            torch.save(net.state_dict(), 'model.pt')
            break

    return scores

scores = train(max_iter=max_iter)
running_mean = [scores[0]]
p = 0.7
for score in scores:
    running_mean.append(p * running_mean[-1] + (1-p) * score)
running_mean = running_mean[1:]

plt.plot(np.arange(len(scores)), scores, label = 'score')
plt.plot(np.arange(len(scores)), running_mean, label = 'running_mean')
plt.xlabel("# Episodes")
plt.ylabel("# Score")
plt.savefig('scores.png')
plt.show()







