#!/usr/bin/env python
import numpy as np
import gym
import random
import sys, copy, argparse
from collections import deque
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, feature_shape, num_actions):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        self.model = Sequential()
        self.model.add(Dense(36, input_shape=feature_shape, activation='relu'))
        self.model.add(Dense(48, activation='relu'))
        # last layer should be linear not relu because we need both negative/positive values
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

    def __init__(self, memory_size=50000, burnin_size=10000):
        self.capacity = memory_size
        self.burnin_size = burnin_size
        self.memory = []

    def sample_batch(self, batch_size=32):
        return random.sample(self.memory, batch_size)

    def append(self, *transaction):
        if len(self.memory) == self.capacity:
            self.memory = self.memory[1:]
        self.memory.append(transaction)


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, mem_size=50000, batch_size=32, tfboard_enabled=False):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.envname = environment_name
        self.env = gym.make(environment_name)
        self.num_actions = self.env.action_space.n
        self.feature_shape = self.env.observation_space.shape
        self.qnet = QNetwork(self.feature_shape, self.num_actions)
        self.model = self.qnet.model
        self.max_iteration = 100000
        self.max_episode = 5000
        self.gamma = 1.
        self.max_eps = 1.
        self.min_eps = .05
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.tfboard_enabled = tfboard_enabled
        self.memory = Replay_Memory()


        # replace default with the recommended parameters........
        if environment_name == 'MountainCar-v0':
            self.max_episode = 5000
            self.gamma = 1.



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
        self.burn_in_memory()
        model = self.model
        tensorboard = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_grads=False, write_images=False)

        # initialization
        total_iteration = 0
        total_reward = 0
        all_reward = []
        for i_episode in range(self.max_episode):

            # reset the environment
            done = False
            curr_state = self.preprocess(self.env.reset())
            curr_iteration = 0
            curr_reward = 0

            # TODO: gym takes care of max_episode_steps??？
            # if one episode reaches max steps, it will be automatically marked as 'done'
            while not done:
                curr_iteration += 1
                # perform eps decay
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
            avg40 = sum(all_reward[-40:]) / min(40, len(all_reward))
            print('----- Episode {} done with {} steps, reward={}, avg40={:.4f}, avg={:.4f} -----'
                  .format(i_episode, curr_iteration, curr_reward, avg40, total_reward/len(all_reward)))
            if i_episode % 100 == 0:
                model.save("./model/{}-e{}-avg40-{:.4f}.h5".format(self.envname, i_episode, avg40))
        model.save("./model/{}-e{}-avg40-{:.4f}.h5".format(self.envname, self.max_episode, sum(all_reward[-40:]) / min(40, len(all_reward))))

    def test(self, model_file=None):
    # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
    # Here you need to interact with the environment, irrespective of whether you are using a memory.
        model = load_model("/Users/cartpole_dqn.h5")
        Test = 100

        for episode in range(self.max_episode):

            # TODO: need a eps-scheduler here. replace following line by eps = some_schedular(episode/time)
            # eps will descent based on the time/episode that already passed


            # reset the environment
            done = False

            if episode % 100 == 0:
                    total_reward = 0
                    for j in range(Test):
                        curr_state = self.env.reset()
                        for i in range(500):
                            curr_state = curr_state[None,:]
                            # calculate current estimated q(s, a) and choose the greedy actio
                            q_curr_estimate = model.predict(curr_state).flatten()
                            # this is the action a
                            curr_action, _ = self.greedy_policy(q_curr_estimate)
                            # take the action to the new state
                            # next_state: s', reward: r
                            next_state, reward, done, _ = self.env.step(curr_action)
                            total_reward+=reward
                            if done:
                                    break
                    average_reward = total_reward/Test
                    #print ('episode: ',episode,'Evaluation Average Reward:',average_reward)
                    if average_reward >= 200:
                            break


    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        print('[burn-in] memory initialization start.')
        episode = 0
        while len(self.memory.memory) < self.memory.burnin_size:
            # beginning of episode
            # only burnin successful episode
            done = False
            transactions = []
            curr_state = self.preprocess(self.env.reset())
            total_reward = 0
            episode += 1
            print('[burn-in] episode={}'.format(episode))
            while not done and len(self.memory.memory) < self.memory.burnin_size:
                action = self.epsilon_greedy_policy(curr_state, self.max_eps)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess(next_state)
                self.memory.append(curr_state, action, reward, next_state, done)
                transactions.append((curr_state, action, reward, next_state, done))
                curr_state = next_state
                total_reward += reward
            if total_reward > -200:
                print('[burn-in] One episode with steps={} reward={} is burned in.'.format(len(transactions), total_reward))
                # for tr in transactions:
                #     self.memory.append(tr)
        print('[burin-in] Done.')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str, default='MountainCar-v0')
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str, default='')
    return parser.parse_args()


def main():
    args = parse_arguments()
    environment_name = args.env

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)
    agent = DQN_Agent('MountainCar-v0')
    agent.train()


# You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main()
