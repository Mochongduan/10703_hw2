#!/usr/bin/env python
import numpy as np
import gym
import sys, copy, argparse
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, num_features, num_actions):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        self.model = Sequential()
        self.model.add(Dense(48, input_shape=(num_features,)))
        self.model.add(Dense(48))
        self.model.add(Dense(num_actions))
        self.model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    def save_model_weights(self, weight_file):
        self.model.save_weights(weight_file)

    def load_model(self, model_name):
        # Helper function to load an existing model.
        self.model = keras.models.load_model(model_name)
        return self.model

    def load_model_weights(self, weight_file):
        self.model.load_weights(weight_file)


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory = deque(maxlen=memory_size)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.epsilon = 1
        for state, action, reward, next_state, done in sample_batch(self)
            target_Q = reward
        if not done:
            target_Q = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
        target_function = self.model.predict(state)
        target_function[0][action] = target_Q
        self.model.fit(state, target_function, epochs=1000, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def sample_batch(self, batch_size=32):
    minibatch = random.sample(self.memory, batch_size)
    return minibatch


def append(self, transition):
    self.memory.append((state, action, reward, next_state, done))


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, render=False, iterations=1000, episodes=1000):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env = gym.make(environment_name)
        self.num_actions = len(self.env.action_space)
        self.num_features = len(self.env.observation_space)
        self.qnet = QNetwork(self.num_features, self.num_actions)

        self.iterations = iterations
        self.episodes = episodes
        self.eps = .9

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        pass

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values, axis=1)

    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        for episodes in range(self.episodes):
            reward = 0
            done = False
            observation = self.env.reset()
            for i in range(self.iterations):



    def test(self, model_file=None):
        pass
    # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
    # Here you need to interact with the environment, irrespective of whether you are using a memory.

    def burn_in_memory():
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    environment_name = args.env

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)


# You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main(sys.argv)
