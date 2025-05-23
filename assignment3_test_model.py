import gym
import simple_driving
import numpy as np
import torch
import torch.nn as nn

# Define the same network structure used during training
class Network(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.layers(x)

# Create environment
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
env = env.unwrapped

# Load trained model
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

model = Network(state_dim, n_actions)
model.load_state_dict(torch.load("dqn_model.pth"))
model.eval()

for i in range (5):
    # Run the agent in the environment
    state, info = env.reset()

    for i in range(200):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state

        if terminated or truncated:
            break
