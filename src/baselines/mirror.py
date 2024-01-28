import numpy as np

from src.baselines.baseline import Baseline


class Mirror(Baseline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lower = -1
        self.higher = -1
    def predict(self, observation, *args, **kwargs):
        if self.lower > observation["carbon_intensity"][0][0] or self.lower == -1:
            self.lower = observation["carbon_intensity"][0][0]
        if self.higher < observation["carbon_intensity"][0][0] or self.higher == -1:
            self.higher = observation["carbon_intensity"][0][0]

        if self.higher == -1 or self.lower == -1 or self.higher == self.lower:
            ess_action = 1
        else:
            ess_action = np.clip((self.lower - observation["carbon_intensity"][0][0]) / ((self.higher-self.lower)/2)+1, -1, 1)
        action = [ess_action] + [0] + [0]
        return self.rescale_action(action, observation, *args, **kwargs)
