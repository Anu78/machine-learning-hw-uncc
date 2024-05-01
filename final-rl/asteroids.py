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

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Defining the convolutional layers
        self.conv_layers = nn.Sequential(
            # Input: 3x160x210
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # Output: 16x160x210
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: 16x80x105
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # Output: 32x80x105
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Output: 32x40x52
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # Output: 64x40x52
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Output: 64x20x26
        )
        
        # Calculating the total number of features after the final pooling layer
        # Output dimensions are 64 channels, each with a size of 20x26
        self.num_features = 64 * 20 * 26
        
        # Defining the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.num_features, 256),  # from calculated features to 256
            nn.ReLU(),
            nn.Linear(256, 14)  # 256 inputs to 14 outputs (actions)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.num_features)  # Manually flatten to ensure correct shape
        x = self.fc_layers(x)
        return x

model = DQN().to(device)
model.load_state_dict(torch.load("./rl-asteroids-v1.pth", map_location=device))
model.eval()

for episode in range(1000):
    observation, info = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state = torch.tensor(observation, dtype=torch.float32, device=device).permute(2,1,0)
        action = torch.argmax(model(state))

        observation, reward, done, _, info = env.step(action.item())
        print(reward)

    print(f"Episode {episode}: Reward = {episode_reward}")

env.close()
