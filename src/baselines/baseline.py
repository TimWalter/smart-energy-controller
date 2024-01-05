from abc import ABC, abstractmethod

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.single_family_home import SingleFamilyHome


class Baseline(ABC):
    def __init__(self, _, env, **kwargs):
        if isinstance(env.unwrapped, DummyVecEnv):
            env = env.unwrapped.envs[0]
        if not isinstance(env, SingleFamilyHome):
            env = env.unwrapped
        self.action_dim = env.action_space.shape[0]
        self.type = env.config["action_space"]["type"]
        self.levels = env.config["action_space"]["levels"]
        self.ess_condition = env.ess_condition
        self.fdr_condition = env.fdr_condition
        self.tcl_condition = env.tcl_condition
        self.resolution = env.resolution
        if self.tcl_condition:
            self.desired_temperature = env.config["thermostatically_controlled_load"]["maximal_temperature"] + \
                                   env.config["thermostatically_controlled_load"]["minimal_temperature"] / 2

    def rescale_action(self, action, observation, *args, **kwargs):
        if not self.ess_condition:
            del action[0]
        if not self.fdr_condition:
            del action[1 if self.ess_condition else 0]
        if not self.tcl_condition:
            del action[-1]
        else:
            tcl_sign = 1 if observation["tcl_indoor_temperature"] < self.desired_temperature else -1
            action[-1] *= tcl_sign
        if self.type == "discrete":
            action = list((np.array(action) + 1 / 2) * (np.array(self.levels) - 1))

        return np.array(action), None

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def learn(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass
