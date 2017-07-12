import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import agent
import enviroment
import history
import layers


class DQN(agent.AgentBase):
    # Modeling
    ENV_ID = 'CartPole-v0'
    STATE_SPACE = 4
    HIDDEN_NEURONS = 20
    ACTION_SPACE = 2
    # Training
    LR = 1e-3  # learning rate

    def __init__(self):
        super().__init__()
        # sub module
        self._env = enviroment.make(self.ENV_ID)
        self._replay = history.ReplayBuffer()
        # build network
        self._net = self._build_network()

    def _build_network(self):
        # neural network
        with tf.name_scope('input'):
            state = tf.placeholder(dtype=tf.float32, shape=[None, self.STATE_SPACE])
        with tf.name_scope('hidden'):
            y1 = layers.fc(state, n_neurons=self.HIDDEN_NEURONS, activation=tf.nn.tanh)
        with tf.name_scope('output'):
            q_predict = layers.fc(y1, n_neurons=self.ACTION_SPACE)
        # process q value
        with tf.name_scope('q_value'):
            action_mask = tf.placeholder(tf.float32, [None, self.ACTION_SPACE])
            q_current = tf.reduce_sum(tf.multiply(q_predict, action_mask), axis=1)
        # loss
        with tf.name_scope('loss'):
            q_target = tf.placeholder(tf.float32, [None])
            loss = tf.reduce_mean(tf.squared_difference(q_current, q_target))
            tf.summary.scalar('loss', loss)
        #
        return {'state': state,
                'action_mask': action_mask,
                'q_current': q_current,
                'q_target': q_target}

    def train(self, episodes=500, max_step=200):
        # prepare for epsilon greedy
        # train step
        for episode in tqdm(range(episodes)):
            if episode % 50 == 0:
                total_reward = self._test_impl(max_step, delay=0.2, gui=True)
                tqdm.write('current reward: {total_reward}'.format(total_reward=total_reward))
            else:
                self._train_impl(max_step)

    def test(self, episodes=1, max_step=200, delay=0.2, gui=True):
        for episode in range(episodes):
            total_reward = self._test_impl(max_step, delay, gui)
            print('current reward: {total_reward}'.format(total_reward=total_reward))

    def _random_action(self, state):
        pass

    def _optimal_action(self, state):
        pass

    def _perceive(self, state, action, state_, reward, done):
        pass
