import gym
import simple_driving
import numpy as np
from collections import defaultdict
import random
import pickle
import math

# Discretize continuous states
def discretize(state, bins=10):
    return tuple((np.array(state) * bins).astype(int))

def default_q():
    return np.zeros(env.action_space.n)

def epsilon_greedy(env, state, Q, epsilon, episodes, episode):
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state])
    else:
        return env.action_space.sample()

def simulate(env, Q, max_episode_length, epsilon, episodes, episode, bins):
    D = []
    state, _ = env.reset()
    state = discretize(state, bins)
    done = False
    for step in range(max_episode_length):
        action = epsilon_greedy(env, state, Q, epsilon, episodes, episode)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = discretize(next_state, bins)
        done = terminated or truncated
        D.append([state, action, reward, next_state])
        state = next_state
        if done:
            break
    return D

def q_learning(env, gamma, episodes, max_episode_length, epsilon, step_size, bins=10):
    # Initialize Q-table
    Q = defaultdict(default_q)
    total_reward = 0
    total_count = 0

    # Training loop
    for episode in range(episodes):
        D = simulate(env, Q, max_episode_length, epsilon, episodes, episode, bins)
        for state, action, reward, next_state in D:
            Q[state][action] = (1 - step_size) * Q[state][action] + step_size * (reward + gamma * np.max(Q[next_state]))
            total_reward += reward
            total_count += 1
            
        # # Decay epsilon
        # epsilon = max(0.1, epsilon * 0.995)

        # Print average reward every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode} | Avg Reward (last 10): {total_reward / total_count:.2f} | Epsilon: {epsilon:.2f}")
            total_reward = 0
            total_count = 0
    return Q

# Setup environment
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
env = env.unwrapped

# Hyperparameters
gamma = 0.9
episodes = 1000
max_episode_length = 200
epsilon = 0.8
step_size = 0.01
bins = 1000  # for discretization

# Train
Q = q_learning(env, gamma, episodes, max_episode_length, epsilon, step_size, bins)

print(f"Q-table contains {len(Q)} states.")

# Save Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)
print("✅ Q-table saved successfully.")

env.close()
