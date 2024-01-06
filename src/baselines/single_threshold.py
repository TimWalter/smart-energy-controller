from src.baselines.baseline import Baseline


class SingleThreshold(Baseline):
    def predict(self, observation, *args, **kwargs):
        if observation["carbon_intensity"] < (0.9 if self.resolution == "minutely" else 65):
            action = [1] + [1] * (self.action_dim - 2) + [0.1]
        elif observation["carbon_intensity"] > 85:
            action = [-1] + [0] * (self.action_dim - 2) + [0]
        else:
            action = [0] + [0] * (self.action_dim - 2) + [0.1]
        return self.rescale_action(action, observation, *args, **kwargs)
