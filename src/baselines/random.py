import numpy as np

from src.baselines.baseline import Baseline


class Random(Baseline):
    def predict(self, *args, **kwargs):
        action = np.random.uniform(-1, 1, max([3, self.action_dim])).tolist()

        return self.rescale_action(action, *args, **kwargs)
