import numpy as np


class SingleThreshold:
    def __init__(self, policy, env, tensorboard_log=None):
        self.action_dim = env.action_space.shape[0]
        self.type = env.unwrapped.config["action_space"]["type"]
        self.levels = env.unwrapped.config["action_space"]["levels"]
        self.energy_storage_system = env.unwrapped.energy_storage_system_condition
        self.flexible_demand_response = env.unwrapped.flexible_demand_response_condition
        self.thermostatically_controlled_load = env.unwrapped.thermostatically_controlled_load_condition

    def predict(self, observation, *args, **kwargs):
        if observation["carbon_intensity"] < 0.9:
            actions = [1, 1, -0.9]
        elif observation["carbon_intensity"] > 1.4:
            actions = [-1, -1, -1]
        else:
            actions = [0, 0, -1]

        if self.type == "discrete":
            actions = list((np.array(actions) + 1 / 2) * (np.array(self.levels) - 1))

        if not self.energy_storage_system:
            del actions[0]
        if not self.flexible_demand_response:
            del actions[1 if self.energy_storage_system else 0]
        if not self.thermostatically_controlled_load:
            del actions[-1]

        return [actions], None

    def learn(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass
