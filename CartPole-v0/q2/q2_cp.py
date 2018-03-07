#!/usr/bin/env python
import argparse
import random

import gym
import keras
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import Sequential


class Replay_Memory():

    def __init__(self, memory_size=50000, burnin_size=10000):
        self.capacity = memory_size
        self.burnin_size = burnin_size
        self.memory = []
        self.pos = 0



    def sample_batch(self, batch_size=32):
        return random.sample(self.memory, batch_size)

    def append(self, *transaction):
        if self.pos == len(self.memory):
            self.memory.append(transaction)
        else:
            self.memory[self.pos] = transaction
        self.pos = (self.pos + 1) % self.capacity


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, mem_size=50000, batch_size=32):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env_name = environment_name
        self.env = gym.make(environment_name)
        # self.env = gym.wrappers.Monitor(self.env, './video/', force=True)
        self.num_actions = self.env.action_space.n
        self.feature_shape = self.env.observation_space.shape
        self.max_iteration = 100000
        self.max_episode = 6000
        self.gamma = .99
        self.max_eps = .5
        self.min_eps = .05
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.period = 10000
        self.memory = Replay_Memory()
        self.model = self.generate_model()


    def generate_model(self):
        model = Sequential()
        model.add(Dense(self.num_actions, input_shape=self.feature_shape))

        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)
        return model


    # This is the phi function described in the paper
    def preprocess(self, state):
        return state[None,:]


    # policy will return (a, q)
    def epsilon_greedy_policy(self, state, eps):
        if np.random.uniform() < eps:
            return self.env.action_space.sample()
        else:
            return self.greedy_policy(state)


    def greedy_policy(self, state):
        # Creating greedy policy for test time.
        return np.argmax(self.model.predict(state))


    def eps_decay(self, i_iteration):
        return min(i_iteration, self.max_iteration) * (self.min_eps - self.max_eps) / self.max_iteration + self.max_eps


    def eval_reward(self, eval_epoch=20):
        total_reward = 0
        for epoch in range(eval_epoch):
            done = False
            curr_state = self.preprocess(self.env.reset())
            while not done:
                action = self.epsilon_greedy_policy(curr_state, self.min_eps)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess(next_state)
                total_reward += reward
                curr_state = next_state
        return total_reward / eval_epoch


    def train_batch(self):
        model = self.model
        batch = self.memory.sample_batch(self.batch_size)
        states = np.zeros((self.batch_size, *self.feature_shape))
        predict = np.zeros((self.batch_size, self.num_actions))
        for i, (curr_state, action, reward, next_state, done) in enumerate(batch):
            states[i] = curr_state
            predict[i] = model.predict(curr_state)
            predict[i][action] = reward if done else (reward + model.predict(next_state).max())
        model.fit(states, predict, verbose=0)


    # policy: callable, which take in q_values and generate actions based on q_values
    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        print('training on {}'.format(self.env_name))
        self.burn_in_memory()
        model = self.model

        # initialization
        total_iteration = 0
        total_reward = 0
        reward_summary =[]
        all_reward = []
        next_iter = 0
        solved = False

        for i_episode in range(self.max_episode):
            # reset the environment
            if solved:
                break

            done = False
            curr_state = self.preprocess(self.env.reset())
            curr_iteration = 0
            curr_reward = 0

            # if one episode reaches max steps, it will be automatically marked as 'done'
            while not done:
                curr_iteration += 1
                eps = self.eps_decay(curr_iteration + total_iteration)
                action = self.epsilon_greedy_policy(curr_state, eps)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess(next_state)
                self.memory.append(curr_state, action, reward, next_state, done)
                self.train_batch()
                # move to the next state
                curr_state = next_state
                curr_reward += reward

            total_iteration += curr_iteration
            total_reward += curr_reward
            all_reward.append(curr_reward)

            # reward summary part!!
            if total_iteration > next_iter:
                next_iter += self.period
                reward_summary.append([total_iteration, self.eval_reward()])

            avg100 = self.latest_k(all_reward)
            print('Episode {} done with {} steps, reward={}, avg100={:.4f}, avg={:.4f} total iter={}'
                  .format(i_episode, curr_iteration, curr_reward, avg100, np.average(all_reward), total_iteration))
            if i_episode % 50 == 0:
                model.save("./model/e{}_avg50_{:.4f}_avg_{}.h5"
                           .format(i_episode, avg100, np.average(all_reward)))
            if avg100 > 195.0:
                solved = True
                print('problem solved with average reward={} for 100 consecutive episodes'.format(avg100))

        np.save('reward_summary.npy', reward_summary)
        model.save("./model/e{}_avg100_{:.4f}_avg_{:4f}.h5"
                   .format(self.max_episode, self.latest_k(all_reward), np.average(all_reward)))


    def latest_k(self, arr, k=100):
        return sum(arr[-k:]) / min(k, len(arr))

    def test(self, model_file=None):
        pass

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        episode = 0
        while len(self.memory.memory) < self.memory.burnin_size:
            # beginning of episode
            # only burnin successful episode
            done = False
            transactions = []
            curr_state = self.preprocess(self.env.reset())
            total_reward = 0
            episode += 1
            while not done and len(self.memory.memory) < self.memory.burnin_size:
                action = self.epsilon_greedy_policy(curr_state, self.max_eps)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess(next_state)
                self.memory.append(curr_state, action, reward, next_state, done)
                transactions.append((curr_state, action, reward, next_state, done))
                curr_state = next_state
                total_reward += reward
        print('[burin-in] Done.')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='CartPole-v0')
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str, default='')
    return parser.parse_args()


def main():
    args = parse_arguments()
    env_name = args.env

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)


    agent = DQN_Agent(env_name)
    agent.train()


# You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main()
