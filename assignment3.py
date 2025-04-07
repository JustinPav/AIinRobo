import gym
import simple_driving
# import pybullet_envs
import pybullet as p
import numpy as np
import math
from collections import defaultdict
import pickle
import torch
import random

def discretize(state, bins=10):
    return tuple((np.array(state) * bins).astype(int))

def default_q():
    return np.zeros(env.action_space.n)

# Load Q-table
with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)

######################### renders image from third person perspective for validating policy ##############################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='tp_camera')
##########################################################################################################################

######################### renders image from onboard camera ###############################################################
# env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True, render_mode='fp_camera')
##########################################################################################################################

######################### if running locally you can just render the environment in pybullet's GUI #######################
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
env = env.unwrapped
##########################################################################################################################

state, info = env.reset()
state = discretize(state)
# frames = []
# frames.append(env.render())

for i in range(200):
    action = np.argmax(Q[state])
    next_state, reward, terminated, truncated, info = env.step(action)
    state = discretize(next_state)
    # frames.append(env.render())  # if running locally not necessary unless you want to grab onboard camera image
    if terminated or truncated:
        break

