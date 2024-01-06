import pickle

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.normalize import RunningMeanStd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


def gather_experiences(agent, env, num_steps):
    obs = env.reset()[0]
    experiences = []
    for _ in range(num_steps):
        action, _ = agent.predict(obs)
        next_obs, reward, done, _, _ = env.step(action[0])
        experiences.append((obs, next_obs, action, reward, done, [{}]))
        obs = next_obs if not done else env.reset()[0]
    return experiences


class NormalizeDictObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        self.obs_rms = RunningMeanStd(shape=len(self.observation_space))
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.normalize(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)
        return self.normalize(obs), info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(np.array([el[0] for el in obs.values()]))
        return {key: (obs[key] - self.obs_rms.mean[i]) / np.sqrt(self.obs_rms.var[i] + self.epsilon) for i, key in
                enumerate(obs.keys())}


class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.infos = []

    def dump(self, path: str):
        with open(path, "wb") as f:
            self.infos = {
                key: {key_inner: [info[key][key_inner] for info in self.infos] for key_inner in
                      self.infos[0][key].keys()}
                if isinstance(self.infos[0][key], dict)
                else [info[key] for info in self.infos]
                for key in self.infos[0].keys()
            }

            pickle.dump(self.infos, f)
        self.infos = []

    def _on_step(self) -> bool:
        self.infos.append(self.locals["infos"][0])
        return True

    def __call__(self, locals_, globals_):
        self.locals = locals_
        self.globals = globals_
        self._on_step()


class TrainCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = {}
        self.infos = []
        self.epoch = 0

    def _on_step(self) -> bool:
        super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.results[f"epoch_{self.epoch}_accumulated_reward"] = self.last_mean_reward
            self.epoch += 1
        else:
            self.infos.append(self.locals["infos"][0])
        return True
