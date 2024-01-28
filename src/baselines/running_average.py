import numpy as np

from src.baselines.baseline import Baseline


class RunningAverage(Baseline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = []
        self.sample_length = 24

    def predict(self, observation, *args, **kwargs):
        self.samples.append(observation["carbon_intensity"][0][0])
        if len(self.samples) > self.sample_length:
            self.samples.pop(0)

        if len(self.samples) < self.sample_length/2:
            return self.rescale_action([0.1, 0, 0], observation, *args, **kwargs)
        average = np.mean(self.samples)
        higher = average + np.std(self.samples) * 3
        lower = average - np.std(self.samples) * 1
        ess_action = np.clip((lower - observation["carbon_intensity"][0][0]) / ((higher-lower)/2)+1, -1, 1)
        action = [ess_action] + [0] + [0]
        return self.rescale_action(action, observation, *args, **kwargs)
