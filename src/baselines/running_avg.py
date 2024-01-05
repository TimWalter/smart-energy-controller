import numpy as np

from src.baselines.baseline import Baseline


class RunningAvg(Baseline):
    def __init__(self, _, env, **kwargs):
        super().__init__(_, env, **kwargs)
        self.window_size = 10
        self.intensities = []

    def predict(self, observation, *args, **kwargs):
        self.intensities += [observation["carbon_intensity"]]
        if len(self.intensities) > self.window_size:
            del self.intensities[0]

        if observation["carbon_intensity"] <= np.mean(self.intensities):
            action = [0.1, 0.2, 0.1]
        else:
            action = [-0.1, -0.5, 0]

        return self.rescale_action(action, observation, *args, **kwargs)
