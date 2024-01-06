from src.baselines.baseline import Baseline


class Idle(Baseline):
    def predict(self, *args, **kwargs):
        action = [0] * self.action_dim
        return self.rescale_action(action, *args, **kwargs)
