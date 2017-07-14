import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import base
import enviroment
import history
import layers


class DQN(base.AgentBase):
    # Modeling
    ENV_ID = 'CartPole-v0'
    STATE_SPACE = 4
    HIDDEN_NEURONS = 20
    ACTION_SPACE = 2
    # Training
    LR = 1e-3  # learning rate
    DF = 0.9

    def __init__(self):
        super().__init__()
        # sub module
        self._env = enviroment.make(self.ENV_ID)
        self._replay_buffer = history.ReplayBuffer()
        # build network
        self._sess = tf.Session()
        self._net = None
        self._target_net = None

    def __del__(self):
        pass

    def _build_network(self):
        pass

    def _build_target_network(self):
        pass