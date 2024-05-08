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
env = gym.make("ALE/Asteroids-v5", difficulty=3, full_action_space=False, obs_type="ram")

# print action space 
print(env.action_space) # 14 possible actions
print(env.observation_space.shape) # 128 bytes of RAM
print(env.action_space.n)  

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


# takes a state from the game and estimates rewards for each action
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 14)   
        )
    
    def forward(self, x):
        return self.fc_layers(x)

# needed to implement epsilon-greedy action selection
# emphasize exploration (picking random actions instead of calling the model) at the start and then gradually shift to exploitation (picking the best action)
def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_model(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# sample a batch from the replay memory and use it to update the model
def optimize_model():
    # ensure we have enough memories
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    # transpose the batch (switch dimensions)
    batch = Transition(*zip(*transitions))

    # compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    # concatenate the batch elements
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_model(state_batch).gather(1, action_batch)

    # compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1).values

    # compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute loss between predicted and expected Q values
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    # clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_value_(policy_model.parameters(), 100)
    optimizer.step()


# hyperparameters
BATCH_SIZE = 256 # how many experiences to sample from
GAMMA = 0.70 # discount factor; how much to value future rewards
EPS_START = 0.2 # what percent of actions are random, at the start
EPS_END = 0.01 # what percent of actions are random, at the end 
EPS_DECAY = 0.000002 # every step, the epsilon value will decay by this amount
TAU = 0.005 # how much to update the target network by
LR = 1e-4 # learning rate for AdamW
UPDATE_TARGET_EVERY = 20000 # how often to update the target network


memory = ReplayMemory(30_000) # replay memory max size
policy_model = DQN().to(device) 
target_model = DQN().to(device)
target_model.load_state_dict(policy_model.state_dict()) # polcy model and target model have the same weights

optimizer = optim.AdamW(policy_model.parameters(), lr=LR, amsgrad=True)
 
# total steps 
steps_done = 0

# helper function to save model in case training is interrupted
def save_models(path):
    torch.save(policy_model.state_dict(), path)
    torch.save(target_model.state_dict(), path)

# start of training loop
num_episodes = 1000

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0) # shape = [128]
    current_lives = info["lives"] # intialize # of lives from environment
    episode_reward = 0 # track rewards per episode
    render = False 

    if i_episode % 20 == 0:
        torch.save(policy_model.state_dict(), "./policy_model.pth")
        torch.save(target_model.state_dict(), "./target_model.pth")

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())

        if info['lives'] < current_lives:
            reward -= 400
            current_lives = info['lives']  

        # penalize staying still (noop = 0)
        if action.item() == 0:
            reward -= 10

        # give 2 for just being alive 
        reward += 2
        episode_reward  += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        if action.dim() == 0:
            print("0 dim action tensor", action)
        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        if steps_done % UPDATE_TARGET_EVERY == 0:
            # update target network using tau
            target_net_state_dict = target_model.state_dict()
            policy_net_state_dict = policy_model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_model.load_state_dict(target_net_state_dict)

        if done:
            print(f"on episode {i_episode}, which lasted {t} frames. reward: {episode_reward}")
            episode_reward = 0
            break

print('Complete')
