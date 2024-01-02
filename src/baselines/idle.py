import numpy as np
from src.environment.single_family_home import SingleFamilyHome
from stable_baselines3.common.vec_env import DummyVecEnv

class Idle:
    def __init__(self, policy, env, **kwargs):
        if isinstance(env.unwrapped, DummyVecEnv):
            env = env.unwrapped.envs[0]
        if not isinstance(env, SingleFamilyHome):
            env = env.unwrapped
        self.action_dim = env.action_space.shape[0]
        self.type = env.config["action_space"]["type"]
        self.levels = env.config["action_space"]["levels"]
        self.energy_storage_system = env.ess_condition
        self.flexible_demand_response = env.fdr_condition
        self.thermostatically_controlled_load = env.tcl_condition

    def predict(self, *args, **kwargs):
        actions = np.zeros((self.action_dim, 1), dtype=np.int64)
        if self.type == "continuous":
            if self.thermostatically_controlled_load:
                actions[-1] = -1
        elif self.type == "discrete":
            if self.energy_storage_system:
                actions[0] = (self.levels[0] - 1)//2
            if self.flexible_demand_response:
                actions[1 if self.energy_storage_system else 0] = (self.levels[1 if self.energy_storage_system else 0] - 1)//2
        return [actions], None

    def learn(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass
