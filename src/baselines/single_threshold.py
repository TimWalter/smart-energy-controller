from src.baselines.baseline import Baseline

import numpy as np
class SingleThreshold(Baseline):
    def predict(self, observation, *args, **kwargs):
        tcl_action = np.clip(np.abs(observation["tcl_indoor_temperature"][0][0] - 20.5)/1.3, 0, 1)
        if observation["carbon_intensity"] < (0.9 if self.resolution == "minutely" else 65):
            action = [1] + [1] * max([1, (self.action_dim - 2)]) + [tcl_action]
        elif observation["carbon_intensity"] > 85:
            action = [-1] + [0] * max([1, (self.action_dim - 2)])+ [tcl_action]
        else:
            action = [0] + [0] * max([1, (self.action_dim - 2)])+ [tcl_action]

        return self.rescale_action(action, observation, *args, **kwargs)
