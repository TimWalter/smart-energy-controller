from datetime import datetime

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.normalize import RunningMeanStd
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


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


class LoggedEvalCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.infos = []

    def _on_step(self) -> bool:
        self.infos.append(self.locals["infos"][0])
        return True

    def __call__(self, locals_, globals_):
        self.locals = locals_
        self.globals = globals_
        self._on_step()


def evaluate_policy_logged(model, env, n_eval_episodes, results, infos):
    eval_callback = LoggedEvalCallback()
    results += [evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, callback=eval_callback)]
    infos += [eval_callback.infos]


class TrainCallback(BaseCallback):
    def __init__(self, results, infos, best_model_save_path, eval_freq, eval_env, n_eval_episodes, verbose=0):
        super().__init__(verbose)
        self.results = results
        self.infos = infos
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.best_model_save_path = best_model_save_path

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            evaluate_policy_logged(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, results=self.results,
                                   infos=self.infos)
            print(f"[{datetime.now()}] Reward: {self.results[-1]}")
        return True
