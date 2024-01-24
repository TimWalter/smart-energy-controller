import json
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType

from environment.components.energy_storage_system import EnergyStorageSystem
from environment.components.external_electricity_supply import ExternalElectricitySupply


class SingleFamilyHome(gym.Env):
    ees: ExternalElectricitySupply
    ess: EnergyStorageSystem

    def __init__(self, config: str):
        self.config = json.load(open(config, "r"))

        self.resolution = self.config["resolution"]
        self.number_of_episodes = 102

        self.ees = ExternalElectricitySupply(resolution=self.resolution)
        self.ess = EnergyStorageSystem(**self.config["energy_storage_system"])

        self.observation_space = self._observation_space()
        self.action_space = self._action_space()

    def _observation_space(self) -> gym.spaces.Dict:
        spaces = {
            "carbon_intensity": gym.spaces.Box(low=0.0512 * 60, high=2.373 * 60),
            "energy_storage_system_charge": gym.spaces.Box(low=0.0, high=self.config["energy_storage_system"][
                "capacity"])
        }
        return gym.spaces.Dict(spaces, seed=42)

    def _construct_observation(self) -> ObsType:
        observation = {
            "carbon_intensity": self.ees.state,
            "energy_storage_system_charge": self.ess.state
        }

        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                observation[key] = value.astype(np.float32)
            else:
                observation[key] = np.array([value], dtype=np.float32)

        return observation

    def _action_space(self) -> gym.spaces.MultiDiscrete | gym.spaces.Box:
        if self.config["action_space"]["type"] == "discrete":
            nvec = [self.config["action_space"]["levels"][0]]
            return gym.spaces.MultiDiscrete(nvec, seed=42)
        elif self.config["action_space"]["type"] == "continuous":
            low = [-1]
            high = [1]
            return gym.spaces.Box(low=np.array(low), high=np.array(high), seed=42)
        else:
            raise ValueError("Invalid action space type.")

    def _rescale_discrete_action(self, action: ActType) -> ActType:
        levels = self.action_space.nvec[0]
        return np.array([2 * action[0] / (levels - 1) - 1], dtype=np.float32).flatten()

    def _calculate_reward(self) -> Tuple[float, dict[str, float]]:
        info = {"ess_reward": self.ees.reward_cache["carbon_intensity"] * (
            -self.ess.reward_cache["consumed_energy"])}
        reward = info["ess_reward"]

        return reward, info

    def _terminal_reward_correction(self) -> Tuple[float, dict[str, float]]:
        reward_correction_info = {"ess_reward": self.ees.reward_cache["carbon_intensity"] * self.ess.state}
        reward_correction = reward_correction_info["ess_reward"]

        return reward_correction, reward_correction_info

    def _calculate_done(self) -> bool:
        return self.ees.time >= self.ees.episode.index[-1]

    def reset(self, seed: int | None = 42, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.ees.reset(episode=0)
        self.ess.reset()

        observation = self._construct_observation()
        return observation, {}

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        if self.config["action_space"]["type"] == "discrete":
            rescaled_action = self._rescale_discrete_action(action)
        else:
            rescaled_action = action
        self.ees.step()

        self.ess.step(rescaled_action[0])

        observation = self._construct_observation()
        reward, reward_info = self._calculate_reward()
        terminated = self._calculate_done()
        truncated = False

        if terminated:
            reward_correction, reward_correction_info = self._terminal_reward_correction()
            reward += reward_correction
            for key, value in reward_correction_info.items():
                reward_info[key] += reward_correction_info[key]

        info = {"next_observation": observation, "action": rescaled_action, "reward": reward,
                "reward_info": reward_info}

        return observation, reward, terminated, truncated, info
