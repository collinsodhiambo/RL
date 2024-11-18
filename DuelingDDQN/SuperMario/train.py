import gym as gym
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from gym.spaces import Box
import gym_super_mario_bros as gsmb
from agent import SuperMarioAgent as net
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import torch
from collections import deque

env = gsmb.make("SuperMarioBros-v0", apply_api_compatibility = True, render_mode = 'human')
env.reset(seed = 72)
env = JoypadSpace(env, COMPLEX_MOVEMENT)

env = ResizeObservation(env, 84)
env = GrayScaleObservation(env)
#env.metadata['video.frames_per_second'] = 120

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc,info
env = SkipFrame(env, skip=4)   
env = FrameStack(env, num_stack=4)

input_dim = env.observation_space.shape
action_dim = env.action_space.n
hidden_dim = 64
lr = 5e-4
buffer_size = int(1e5)
batch_size = 32
TAU = 1e-2
GAMMA = 0.99
CLIP = 10.0
LEARN_EVERY = 4
seed = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
eps_decay = 0.95

net = net(input_dim=input_dim,action_dim=action_dim,hidden_dim=hidden_dim,lr=lr,
          buffer_size=buffer_size,batch_size=batch_size,TAU=TAU,GAMMA=GAMMA,
          WEIGHT_NORM_CLIP=CLIP,LEARN_EVERY=LEARN_EVERY,seed=seed,device=device)
print("Model parameters: ", sum([p.numel() for p in net.local_model.parameters()])/1e6)



def train(episodes = 1, max_iter = 5000, eps = 0.01):
    #/kaggle/input/ddqn9/pytorch/default/1/
    ckpt = torch.load("net_checkpoint_episodes_301.pt", map_location = device, weights_only = True)
    net.local_model.load_state_dict(ckpt['model_state_dict'])
    net.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print("[INFO]: Training...")
    scores = []
    scores_window = deque(maxlen=100)
    episode_len = deque(maxlen=100)
    

    for i_episode in range(300, episodes):
        state = env.reset()
        total_reward = 0
        state, info = env.reset()

        for k in range(max_iter):
            #env.render()
            action = net.act(np.array(state), eps)
            next_state, reward, done, trunc,info = env.step(action)
            net.step(np.array(state), action, reward, np.array(next_state), done)
            total_reward += reward
            if (done or trunc):
                break
            state = next_state
        scores_window.append(total_reward)
        scores.append(total_reward)
        episode_len.append(k)
        eps = max(0.01, eps * eps_decay)

        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {}\tAverage episode length: {:.2f}'.format(i_episode, np.mean(scores_window), eps, np.mean(episode_len)), end="")
        if i_episode % 50 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {}\tAverage episode length: {:.2f}'.format(i_episode, np.mean(scores_window), eps, np.mean(episode_len)))

            data = {"model_state_dict" : net.local_model.state_dict(),
                    "optimizer_state_dict" : net.optimizer.state_dict()}
            name = f"net_checkpoint_episodes_{i_episode}.pt"
            torch.save(data, name)
    env.close()
            
    return scores

def play(n = 1):
    ckpt = torch.load("net_checkpoint_episodes_650.pt", map_location=device, weights_only = True)
    net.local_model.load_state_dict(ckpt["model_state_dict"])
    for i in range(n):
        total_reward = 0.0
        state, info = env.reset()
        score = 0
        env.render()
        while True:
            action = net.act(np.array(state))
            state, reward, done, trunc, info = env.step(action)
            #print(info)
            
            #new_score = info['score']
            #if new_score != score:
             #   score = new_score
                
            #if info['flag_get']:
                #score += 500
            
            #reward += score
            total_reward += reward
            if (done or trunc):
                break
        
        print(total_reward)
    env.close()

play(2)
import code
code.interact(local=locals())
