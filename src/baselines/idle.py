from src.baselines.baseline import Baseline


class Idle(Baseline):
    def predict(self, *args, **kwargs):
        action = [0, 0, 0]
        return self.rescale_action(action, *args, **kwargs)
