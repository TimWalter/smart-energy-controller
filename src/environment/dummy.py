from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType

from stable_baselines3.common.callbacks import BaseCallback


class LoggingCallback(BaseCallback):
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


class Dummy(gym.Env):
    def __init__(self):

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,),dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.state = 0.5
        self.counter = 0
        self.coefficient = 0.4

    def reset(self, seed: int | None = 42, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        episode = 0
        self.counter = 0

        self.state = 0.5

        observation = np.array([self.state], dtype=np.float32)
        return observation, {}

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        self.state += action[0] * 0.1

        observation = np.array([self.state], dtype=np.float32)
        reward = self.state - action[0] * self.coefficient
        terminated = self.counter > 100
        truncated = False
        self.counter += 1

        info = {"observation": observation, "action": action, "reward": reward}

        return observation, reward, terminated, truncated, info

