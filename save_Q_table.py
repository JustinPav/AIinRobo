import gym
import simple_driving
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt

# Hyperparameters
FC1_DIMS = 256
FC2_DIMS = 256
LEARNING_RATE = 0.0005
GAMMA = 0.99
MEM_SIZE = 100000
MEM_RETAIN = 0.1
BATCH_SIZE = 64
REPLAY_START_SIZE = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 5000
NETWORK_UPDATE_ITERS = 1000
EPISODES = 1000

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.n

        self.layers = nn.Sequential(
            nn.Linear(self.input_shape[0], FC1_DIMS),
            nn.ReLU(),
            nn.Linear(FC1_DIMS, FC2_DIMS),
            nn.ReLU(),
            nn.Linear(FC2_DIMS, self.action_space)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.layers(x)

class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape), dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=bool)

    def add(self, state, action, reward, state_, done):
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count
        else:
            mem_index = int(self.mem_count % ((1 - MEM_RETAIN) * MEM_SIZE) + (MEM_RETAIN * MEM_SIZE))

        self.states[mem_index] = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] = 1 - done

        self.mem_count += 1

    def sample(self):
        max_mem = min(self.mem_count, MEM_SIZE)
        indices = np.random.choice(max_mem, BATCH_SIZE, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.states_[indices],
            self.dones[indices],
        )

class DQN_Solver:
    def __init__(self, env):
        self.memory = ReplayBuffer(env)
        self.policy_network = Network(env)
        self.target_network = Network(env)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.learn_count = 0
        self.env = env

    def choose_action(self, state):
        if self.memory.mem_count > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.learn_count / EPS_DECAY)
        else:
            eps_threshold = 1.0

        if random.random() < eps_threshold:
            action_distribution = [0.1, 0.2, 0.1, 0.09, 0.02, 0.09, 0.1, 0.2, 0.1]
            return np.random.choice(np.arange(9), p=action_distribution)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_network(state_tensor)
        return torch.argmax(q_values).item()

    def learn(self):
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        states_ = torch.tensor(states_, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        q_pred = self.policy_network(states)[np.arange(BATCH_SIZE), actions]

        with torch.no_grad():
            q_next = self.target_network(states_)
            q_target = rewards + GAMMA * torch.max(q_next, dim=1)[0] * dones

        loss = self.policy_network.loss(q_pred, q_target)

        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()

        self.learn_count += 1
        if self.learn_count % NETWORK_UPDATE_ITERS == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())


# ========== Training ==========
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
env = env.unwrapped

agent = DQN_Solver(env)
episode_rewards = []

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    while True:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        # Optional: add reward shaping here
        agent.memory.add(state, action, reward, next_state, done)

        if agent.memory.mem_count > REPLAY_START_SIZE:
            agent.learn()

        state = next_state
        total_reward += reward
        if done:
            break

    episode_rewards.append(total_reward)
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")

# Plot training performance
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training Performance")
plt.show()

torch.save(agent.policy_network.state_dict(), "dqn_model.pth")

