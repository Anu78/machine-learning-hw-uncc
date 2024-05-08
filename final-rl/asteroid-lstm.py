#!/usr/bin/env python
# coding: utf-8
import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


# increase difficulty to penalize the ai for staying still (more asteroids)
env = gym.make("ALE/Asteroids-v5", difficulty=2, full_action_space=False, obs_type="rgb")

# print action space 
print(env.observation_space.shape) # print size of state
print(env.action_space.n) # print number of actions

# cuda or mps
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# this is what's stored in the replay memory every frame
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# store experiences and sample them randomly to train the model
class ReplayMemory(object):
    def __init__(self, capacity):
        # use a deque to cap the length of experiences
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# feature extraction to get: ship location, asteroid locations

# takes many previous states from the game and produces rewards for each action
# an lstm is used to help the model understand motion between frames
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)  

        self.lstm = nn.LSTM(512, 256, batch_first=True)  

        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        
        x = x.view(batch_size, timesteps, -1)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  
        
        x = self.fc2(x)
        return x

# print number of trainable parameters
def count_parameters(model): # 2,474,158 parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# needed to implement epsilon-greedy action selection
# emphasize exploration (picking random actions instead of calling the model) at the start and then gradually shift to exploitation (picking the best action)
def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return torch.tensor([policy_model(state).argmax()], device= device)
    else:
        return torch.tensor([env.action_space.sample()], device=device, dtype=torch.uint8)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state if s is not None], dim=0)

    state_batch = torch.cat([s.unsqueeze(0) for s in batch.state], dim=0)
    action_batch = torch.cat(batch.action).unsqueeze(1)  # Ensure this is a column vector
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_model(state_batch).gather(1, action_batch).squeeze(1)  

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if len(non_final_next_states) > 0:
            next_state_values[non_final_mask] = target_model(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 100)
    optimizer.step()


# hyperparameters
BATCH_SIZE = 256 # how many experiences to sample from
GAMMA = 0.7 # discount factor; how much to value future rewards
EPS_START = 0.2 # what percent of actions are random, at the start
EPS_END = 0.01 # what percent of actions are random, at the end 
EPS_DECAY = 2e-5 # every step, the epsilon value will decay by this amount | ends at 95,000 steps
TAU = 5e-3 # how much to update the target network by
LR = 1e-4 # learning rate for AdamW
UPDATE_TARGET_EVERY = 20_000 # how often to update the target network
LSTM_CONTEXT = 5 # how many previous frames to consider when making a decision


memory = ReplayMemory(10_000) # replay memory max size
policy_model = DQN().to(device) 
target_model = DQN().to(device)


target_model.load_state_dict(policy_model.state_dict()) # policy model and target model have the same weights at initialization

optimizer = optim.AdamW(policy_model.parameters(), lr=LR, amsgrad=True)
 
# total steps 
steps_done = 0

# helper function to save model in case training is interrupted
def save_models(path):
    torch.save(policy_model.state_dict(), path)
    torch.save(target_model.state_dict(), path)

# start of training loop
num_episodes = 1000
render = True

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state,dtype=torch.float32,device=device).permute(2,1,0)
    current_lives = info["lives"] # intialize # of lives from environment
    frame_count = 0 # to skip frames
    intermediate_frames = [] # store frames to be used in LSTM

    # save model every 20 episodes
    if i_episode % 20 == 0:
        torch.save(policy_model.state_dict(), "./policy_model.pth")
        torch.save(target_model.state_dict(), "./target_model.pth")

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())

        if info['lives'] < current_lives:
            reward -= 300 # significant penalty for losing a life (getting hit by an asteroid)
            current_lives = info['lives']  

        reward += 2 # reward for staying alive every frame
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).permute(2,1,0)

        if action.dim() == 0:
            print("0 dim action tensor", action)
        
        # push intermediate frames to memory
        intermediate_frames.append((state, action, next_state, reward))

        if frame_count % LSTM_CONTEXT == 0 and frame_count != 0:
            memory.push([frame for frame in intermediate_frames])
            intermediate_frames.clear()

        state = next_state

        optimize_model()

        # soft update target network using tau
        if steps_done % UPDATE_TARGET_EVERY == 0:
            for target_param, source_param in zip(target_model.parameters(), policy_model.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - TAU) + source_param.data * TAU)

        if done:
            print(f"on episode {i_episode}, which lasted {t} frames")
            break

print('Complete')
