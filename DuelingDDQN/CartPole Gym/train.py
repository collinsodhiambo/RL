import gymnasium as gym
from agent import Agent
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.manual_seed(42)

env = gym.make("CartPole-v1")
state = env.reset(seed=42)

state_dim = 4
hidden_dim = 64
action_dim = env.action_space.n

net = Agent(state_dim, hidden_dim, action_dim)
print("Model parameters: ", sum([p.numel() for p in net.local_model.parameters()]))

episodes = 2000
max_iter = 1000
scores = [] # for plotting
scores_window = deque(maxlen=100) # holds the last 200 rewards
eps = 1.0
eps_decay = 0.99

#f = open("training_logs2.txt", "w")
def train(episodes = 2000, max_iter = 1000):
    scores = [] # for plotting
    scores_window = deque(maxlen=100) # holds the last 200 rewards
    eps = 1.0
    eps_decay = 0.99

    f = open("training_logs.txt", "w")
    for i_episode in range(1, 1 + episodes):
        state = env.reset()
        score = 0
        for it in range(max_iter):
            action = net.act(state[0] if isinstance(state, tuple) else state, eps)
            next_state, reward, done, _, _ = env.step(action)
            net.step(state[0] if isinstance(state, tuple) else state, action, reward, next_state[0] if isinstance(next_state, tuple) else next_state, done)
            score += reward
            if done:
                break
            state = next_state
        scores.append(score)
        scores_window.append(score)
        eps = max(0.01, eps * eps_decay)

        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {}'.format(i_episode, np.mean(scores_window), eps), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {}'.format(i_episode, np.mean(scores_window), eps))
            f.write('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {}'.format(i_episode, np.mean(scores_window), eps))
        if np.mean(scores_window)>=195.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            data = {"model_state_dict" : net.local_model.state_dict(),
                    "optimizer_state_dict" : net.optimizer.state_dict()}
            name = f"net_checkpoint_episodes_{1 + i_episode}_2.pt"
            torch.save(data, name)
            f.write('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            f.close()
            break
    return scores
scores = train()
x = np.arange(len(scores))
y = scores

plt.plot(x, y)
plt.xlabel("Number of Episodes")
plt.ylabel("Agent score") 
plt.savefig("score_vs_episodes.png")
plt.show()
