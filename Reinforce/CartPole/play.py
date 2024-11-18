import torch
from model import Policy as PG
import gymnasium as gym

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('reinforce_472_steps.pt', weights_only=True,
                  map_location=device)['model_state_dict']

env = gym.make('CartPole-v1', render_mode = 'human')
env.reset(seed=35)


n_actions = env.action_space.n
input_dim = env.observation_space.shape[0]
hidden_dim = 16

net = PG(input_dims=input_dim, hidden_dim=hidden_dim,
         n_actions=n_actions,device=device, seed=42)
net.load_state_dict(ckpt)

def play(n=5):
    for i in range(n):
        state, _ = env.reset()
        total = 0.0
        while True:
            action, _ = net.act(state)
            state, reward, done, trunc, _ = env.step(action)
            total += reward

            if done or trunc:
                break

        print(total)
    env.close()

play()
