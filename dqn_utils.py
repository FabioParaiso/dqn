import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import numpy as np
import random

class model:
    def __init__(self, num_states, num_action):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape= (num_states,)))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(num_action, activation='linear'))
        self.model.compile(optimizer='adam', loss = 'mse', metrics=['mae'])

    def get_action(self, state, epsilon):
        if np.random.random() > epsilon:
           action = np.argmax(self.model.predict(state.reshape(-1, 4)))
        else:
            action = np.random.randint(0,2)
        return action
    
    def get_q(self, state):
        return self.model.predict(state.reshape(-1, 4))
    
    def get_max_q(self, state):
        return np.max(self.model.predict(state.reshape(-1, 4)))
    
    def train(self, X, y):
        self.model.fit(X, y, epochs=1, verbose=0)
        
    def save_model(self):
        self.model.save('DQN.model')


class train_data:
    def __init__(self):
        self.min_samples = 1_000
        self.max_samples = 5_000
        self.batch_size = 64
        self.data = []
        
    def train_model(self, train): 
        if len(self.data) < self.min_samples:
            return
        else:
            X, y = self.get_batch(train)
            train.train(X, y)
            
    
    def get_batch(self, train):
        DISCOUNT = 0.95
        
        samples = random.sample(self.data, self.batch_size)
        
        X = []
        y = []
        
        for sample in samples:
            state = sample[0]
            action = sample[1]
            reward = sample[2]
            new_state = sample[3]
            done = sample[4]
            
            # X is pretty simple
            # [0] is the state passed from the episode
            X.append(state)
            
            # For the y we have to make some calculations
            # First the current q using the train model
            current_q = train.get_q(state)

            # And then the target q using the reward, discount and max future q
            # However, if the episode ended on this step we just consider the reward (negative because it lost)
            if done: 
                target_q = -200
            else:
                max_future_q = train.get_max_q(new_state)
                target_q = reward + DISCOUNT * max_future_q
                
            # We then correct the current q for the action take with the target q
            current_q[0][action] = target_q
            
            # And finally we append our data to our y list
            y.append(current_q)

        return np.array(X), np.array(y)
    
    def storage_data(self, results):
        if len(self.data) >= self.max_samples:
            self.data.pop(0)
        self.data.append(results)
        
        
        
        
        