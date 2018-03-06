#!/usr/bin/env python
import numpy as np
import gym
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
        self.model.add(Dense(num_actions, input_shape=feature_shape))
        self.model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mse)

    def save_model_weights(self, weight_file):
        self.model.save_weights(weight_file)

    def load_model(self, model_name):
        # Helper function to load an existing model.
        self.model = keras.models.load_model(model_name)
        return self.model

    def load_model_weights(self, weight_file):
        self.model.load_weights(weight_file)


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, render=False, gamma=.99,
                 max_iteration=100000, max_episode=5000,
                 max_eps=.5, min_eps=.05,
                 mem_size=50000, batch_size=32, tfboard_enabled=False):
        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env = gym.make(environment_name)
        self.num_actions = self.env.action_space.n
        self.feature_shape = self.env.observation_space.shape
        self.qnet = QNetwork(self.feature_shape, self.num_actions)

        self.max_iteration = max_iteration
        self.max_episode = max_episode
        self.gamma = gamma
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.tfboard_enabled = tfboard_enabled

        # replace default with the recommended parameters........
        if environment_name == 'MountainCar-v0':
            self.max_episode = 5000
            self.gamma = 1.
        elif environment_name == 'CartPole-v0':
            self.max_episode = 1000000
            self.gamma = .99


    # policy will return (a, q)
    def epsilon_greedy_policy(self, q_values, eps):
        # Creating epsilon greedy probabilities to sample from.
        # TODO: check eps-greedy policy
        if np.random.uniform() < eps:
            a = self.env.action_space.sample()
            return a, q_values[0][a]
        else:
            return self.greedy_policy(q_values)


    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        a = q_values[0].argmax()
        return a, q_values[0][a]


    def eps_decay(self, i_iteration):
        return min(i_iteration, self.max_iteration) * (self.min_eps - self.max_eps) / self.max_iteration + self.max_eps


    # policy: callable, which take in q_values and generate actions based on q_values
    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        model = self.qnet.model
        tensorboard = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_grads=False, write_images=False)
        total_iteration = 0

        for i_episode in range(self.max_episode):
            # exceed the maximum iteration limit
            if total_iteration > self.max_iteration:
                break

            # reset the environment
            done = False
            curr_state = self.env.reset()
            curr_iteration = 0
            total_reward = 0

            # TODO: gym takes care of max_episode_steps??？
            # if one episode reaches max steps, it will be automatically marked as 'done'
            while not done:
                curr_iteration += 1
                eps = self.eps_decay(curr_iteration + total_iteration)

                # TODO: currently using batch_size = 1, change to mini-batch later
                curr_state = curr_state[None,:]

                # calculate current estimated q(s, a) and choose the greedy action
                q_curr_target = model.predict(curr_state)

                # this is the action a
                curr_action, _ = self.epsilon_greedy_policy(q_curr_target, eps)

                # take the action to the new state
                # next_state: s', reward: r
                next_state, reward, done, _ = self.env.step(curr_action)
                total_reward += reward

                if done:
                    q_next = 0
                else:
                    # this is q(s', a')
                    q_next_estimate = model.predict(next_state[None,:])

                    # TODO: epsilon-greedy or greedy?
                    _, q_next = self.greedy_policy(q_next_estimate)

                # r + gamma * max(q(s', a'))
                q_curr_target[0][curr_action] = reward + self.gamma * q_next

                # q_curr_target = keras.utils.to_categorical([[q_curr_target]], self.num_actions)
                # train it
                if self.tfboard_enabled:
                    model.fit(curr_state, q_curr_target, verbose=0, callbacks=[tensorboard])
                else:
                    model.fit(curr_state, q_curr_target, verbose=0)
                # move to the next state
                curr_state = next_state

            print('----- Episode {} done with {} steps, total reward={} -----'.format(i_episode, curr_iteration, total_reward))
            total_iteration += curr_iteration
            if i_episode % 500 == 0:
                model.save("./model/cartpole_dqn{}.h5".format(i_episode))


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
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
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
    agent = DQN_Agent(environment_name)
    agent.train()


# You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main()
