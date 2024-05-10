#!/opt/homebrew/bin/python3.11
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
import matplotlib.pyplot as plt

# increase difficulty to penalize the ai for staying still (more asteroids)
env = gym.make("ALE/Asteroids-v5", difficulty=3, full_action_space=False, obs_type="rgb")

# print action space 
print(env.observation_space.shape) # print size of state
print(env.action_space.n) # print number of actions

# cuda or mps (apple)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# this is what's stored in the replay memory every frame
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# store experiences and sample them randomly to train the model
class ReplayMemory:
    def __init__(self, capacity):
        # use a deque to cap the length of experiences
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        samples = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*zip(*samples))
        return batch

    def __len__(self):
        return len(self.memory)

# takes n previous states from the game and produces rewards for each action
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
def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# calculate moving average of final data
def moving_average(x, w):
    weights = np.ones(w) / w
    return np.convolve(x, weights, mode='valid')

# implementation of epsilon-greedy action selection
# emphasize exploration (picking random actions instead of calling the model) at the start and then gradually shift to exploitation (picking the best action)
def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_model(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([env.action_space.sample()], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state if s is not None], dim=0)

    state_batch = torch.cat([s.unsqueeze(0) for s in batch.state], dim=0)
    action_batch = torch.cat(batch.action).unsqueeze(1)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_model(state_batch).gather(1, action_batch).squeeze(1)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

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
EPS_START = 0.3 # what percent of actions are random, at the start
EPS_END = 0.01 # what percent of actions are random, at the end 
EPS_DECAY = 2e-5 # every step, the epsilon value will decay by this amount | ends at 95,000 steps
TAU = 5e-3 # how much to update the target network by
LR = 1e-4 # learning rate for AdamW
UPDATE_TARGET_EVERY = 20_000 # how often to update the target network
LSTM_CONTEXT = 5 # how many previous frames to consider when making a decision
memory = ReplayMemory(10_000) # replay memory max size
policy_model = DQN(6).to(device) # reduced action space (essential controls only)
target_model = DQN(6).to(device)
target_model.load_state_dict(policy_model.state_dict()) # policy model and target model have the same weights at initialization
optimizer = optim.AdamW(policy_model.parameters(), lr=LR, amsgrad=True)
steps_done = 0 # all time step count

# helper function to save model in case training is interrupted
def save_models():
    torch.save(policy_model.state_dict(), "./policy.pth")
    torch.save(target_model.state_dict(), "./target.pth")

# start of training loop
num_episodes = 1000 
render = False # used to render every n episodes
episode_rewards = np.zeros(num_episodes) # all time rewards
episode_lengths = np.zeros(num_episodes) # all time lengths

try:
    for ep in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
        print(state.shape)
        current_lives = info["lives"]
        state_buffer = deque([state for _ in range(LSTM_CONTEXT)], maxlen=LSTM_CONTEXT)
        render = ep % 20 == 0 # live progress every 20 episodes

        for t in count():
            state_seq = torch.cat(list(state_buffer), dim=1)
            action = select_action(state_seq)
            observation, reward, terminated, truncated, info = env.step(action.item())

            if render:
                env.render()
            
            # penalize agent if it gets hit
            reward_base = 2 - 250 * (info["lives"] < current_lives) 
            current_lives = info["lives"]

            episode_rewards[ep] += reward_base
            reward = torch.tensor([reward_base], device=device)
            done = terminated or truncated
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0) if not terminated else None

            state = next_state

            state_buffer.append(state)

            # push to memory every LSTM_CONTEXT frames
            # memory now contains a sequence of states
            if t % LSTM_CONTEXT == 0:
                state_seq = torch.cat(list(state_buffer), dim=1)
                memory.push(state_seq, action, next_state, reward)

            optimize_model()

            # soft update the target network
            if steps_done % UPDATE_TARGET_EVERY == 0:
                for target_param, source_param in zip(target_model.parameters(), policy_model.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - TAU) + source_param.data * TAU)

            if done:
                print(f"episode {ep} | {t} frames | reward: {episode_rewards[ep]}")
                episode_lengths[ep] = t
                break

except KeyboardInterrupt:
    save_models()
    print("Training interrupted; models saved. Displaying plots.")
    # plot rewards and length
    plt.plot(episode_rewards)
    plt.plot(moving_average(episode_rewards, 30))
    plt.show()
    plt.plot(episode_lengths)
    plt.plot(moving_average(episode_lengths, 30))
    plt.show()

    print('Complete')