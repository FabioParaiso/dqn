import gym
import numpy as np
from dqn_utils import model, train_data
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Calling the gym enviornment
env = gym.make('CartPole-v1')

# Initialization of the model class
trainNN = model(4, 2)

# Initialization of the class to storage and train the model
td = train_data()

# Epsilon configuration
epsilon = 1
EPSILON_DECAY = 0.99

# Maximum number of episodes
MAX_EPISODES = 5000

# List of episode rewards
rewards = []

# Sets at which episodes we should update the target model
MODEL_UPDATE_EPISODE = 5

# Cycle through the episodes
for episode in range(MAX_EPISODES):

    # Resets the episode reward
    i = 0

    # Resets the enviornment variables
    done = False
    state = env.reset()
    
    while not done:
    
        # Get action
        action = trainNN.get_action(state, epsilon)
        
        # New state
        new_state, reward, done, _ = env.step(action)
        
        if not done or i == env._max_episode_steps-1:
            reward= reward
        else:
            reward = -200
        
        # Storages the state info
        state_info = [state, action, reward, new_state, done]
        td.storage_data(state_info)
        
        # Things to do at the end of the episode
        epsilon *= EPSILON_DECAY
        
        # Training the train model
        td.train_model(trainNN)
        
        # Things to do at the end of an action
        i += 1
        state = new_state
        
    #if episode % 10 == 0:
    print(f"Episode {episode} of {MAX_EPISODES} with and average reward of {i} and epsilon of {epsilon})")
    