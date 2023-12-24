import numpy as np


class Idle:
    def __init__(self, policy, env, tensorboard_log):
        self.action_dim = env.action_space.shape[0]

    def predict(self, *args, **kwargs):
        return np.zeros((self.action_dim, 1), dtype=np.int64), None

    def learn(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass