# test script to eventually play asteroids 
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("mps")

env = gym.make("ALE/Asteroids-v5", render_mode="human")

print(env.observation_space.shape)
print(env.action_space.n)

class AsteroidsCNN(nn.Module):
    def __init__(self):
        super(AsteroidsCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Input: 210x160x3
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # Output: 210x160x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: 105x80x16
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # Output: 105x80x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: 52x40x32
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # Output: 52x40x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Output: 26x20x64
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flatten the output of the last pooling layer
            nn.Linear(26*20*64, 256),     # 26*20*64 inputs, 256 outputs
            nn.ReLU(),
            nn.Linear(256, 14)            # 256 inputs, 14 outputs (actions)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

for episode in range(1000):
    observation, info = env.reset()
    episode_reward = 0
    done = False

    while not done:
        print(observation.shape)

        observation, reward, done, _, _= env.step(env.action_space.sample())

    print(f"Episode {episode}: Reward = {episode_reward}")

env.close()
