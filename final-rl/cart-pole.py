import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("mps")

class CartPoleModel(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(CartPoleModel, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x) # No activation on the output

model = CartPoleModel(4,2).to(device)
model.load_state_dict(torch.load("./target.pth"))
model.eval()
env = gym.make("CartPole-v1", render_mode="human")
num_episodes = 10000
GAMMA = 0.99
BATCH_SIZE = 32
EPSILON = 0.1  # Exploration probability
BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000
memory = []

for episode in range(num_episodes):
    observation, info = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        action = torch.argmax(model(state))

        observation, reward, done, _, _= env.step(action.item())

    print(f"Episode {episode}: Reward = {episode_reward}")

env.close()
