# test script to eventually play asteroids 
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("mps")

env = gym.make("ALE/Asteroids-v5", render_mode="human", difficulty=3, full_action_space=False, obs_type="ram")

print(env.observation_space.shape)
print(env.action_space.n)

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

model = DQN().to(device)
model.load_state_dict(torch.load("./policy_model.pth", map_location=device))
model.eval()

for episode in range(1):
    observation, info = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state = torch.tensor(observation, dtype=torch.float32, device=device)
        action = model(state).argmax()
        print(action)

        observation, reward, done, _, info = env.step(action.item())
        env.render()

    print(f"Episode {episode}: Reward = {episode_reward}")