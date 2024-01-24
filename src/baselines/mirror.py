from src.baselines.baseline import Baseline
import numpy as np

class Mirror(Baseline):
    def predict(self, observation, *args, **kwargs):
        ess_action = np.clip((100 - observation["carbon_intensity"][0][0]) / 35 - 1, -1, 1)
        action = [ess_action] + [0] + [0]
        return self.rescale_action(action, observation, *args, **kwargs)
