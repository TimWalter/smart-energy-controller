import numpy as np


class Idle:
    def __init__(self, policy, env, **kwargs):
        self.action_dim = env.action_space.shape[0]
        self.type = env.unwrapped.config["action_space"]["type"]
        self.levels = env.unwrapped.config["action_space"]["levels"]
        self.energy_storage_system = env.unwrapped.energy_storage_system_condition
        self.flexible_demand_response = env.unwrapped.flexible_demand_response_condition
        self.thermostatically_controlled_load = env.unwrapped.thermostatically_controlled_load_condition

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
