from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType
from stable_baselines3.common.callbacks import BaseCallback

from environment.components.adaptive_consumption import AdaptiveConsumption, AdaptiveConsumptionParameters
from environment.components.battery import Battery, BatteryParameters
from environment.components.consumption import Consumption
from environment.components.electricity_grid import ElectricityGrid
from environment.components.generation import Generation
from environment.components.information import Information
from environment.components.tcl import TCL, TCLParameters


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


class SingleFamilyHome(gym.Env):
    def __init__(self,
                 adaptive_consumption_params: AdaptiveConsumptionParameters = AdaptiveConsumptionParameters(),
                 battery_params: BatteryParameters = BatteryParameters(),
                 tcl_params: TCLParameters = TCLParameters(),
                 include_infos=True,
                 synthetic_data: bool = False,
                 episode_length: int = None,
                 normalize_observation: bool = True):

        self.adaptive_consumption_params = adaptive_consumption_params
        self.battery_params = battery_params
        self.tcl_parameters = tcl_params
        self.include_infos = include_infos
        self.synthetic_data = synthetic_data
        self.episode_length = episode_length
        self.normalize_observation = normalize_observation

        self.action_space_shape = 0
        self.observation_space_shape = 0

        self.action_slices = {}
        self.observation_slices = {}

        self.components = {}

        self.add("intensity", 1, observation=True, action=False, component=False)
        self.add("generation", 1, observation=True, action=False, component=True)
        self.add("consumption", 1, observation=True, action=False, component=True)

        if self.adaptive_consumption_params is not None:
            self.add("adaptive_consumption", self.adaptive_consumption_params.planning_horizon * 2 + 1,
                     observation=True, action=True, component=True)

        if self.battery_params is not None:
            self.add("battery", 1, observation=True, action=True, component=True)

        if self.tcl_parameters is not None:
            self.add("tcl", 1, observation=True, action=True, component=True)

        if include_infos:
            self.add("day_of_year", 1, observation=True, action=False, component=False)
            self.add("hour_of_day", 1, observation=True, action=False, component=False)
            self.add("solar_irradiation", 1, observation=True, action=False, component=False)
            self.add("solar_elevation", 1, observation=True, action=False, component=False)
            self.add("temperature", 1, observation=True, action=False, component=False)
            self.add("wind_speed", 1, observation=True, action=False, component=False)
            self.add("information", 1, observation=False, action=False, component=True)

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.observation_space_shape,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_space_shape,), dtype=np.float32)

    def add(self, name: str, dim: int, observation: bool, action: bool, component: bool):
        if observation:
            self.observation_space_shape += dim
            self.observation_slices[name] = slice(self.observation_space_shape - dim, self.observation_space_shape)
        if action:
            self.action_space_shape += dim
            self.action_slices[name] = slice(self.action_space_shape - dim, self.action_space_shape)
        if component:
            self.components[name] = None

    def reset(self, seed: int | None = 42, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        episode = 0

        self.components["intensity"] = ElectricityGrid(episode=episode,
                                                       synthetic_data=self.synthetic_data,
                                                       episode_length=self.episode_length,
                                                       normalise=self.normalize_observation)

        self.components["generation"] = Generation(episode=episode,
                                                   synthetic_data=self.synthetic_data,
                                                   episode_length=self.episode_length,
                                                   normalise=self.normalize_observation)

        self.components["consumption"] = Consumption(episode=episode,
                                                     synthetic_data=self.synthetic_data,
                                                     episode_length=self.episode_length,
                                                     normalise=self.normalize_observation)

        if self.adaptive_consumption_params is not None:
            self.components["adaptive_consumption"] = AdaptiveConsumption(**self.adaptive_consumption_params.__dict__,
                                                                          episode=episode,
                                                                          synthetic_data=self.synthetic_data,
                                                                          episode_length=self.episode_length,
                                                                          normalise=self.normalize_observation)

        if self.battery_params is not None:
            self.components["battery"] = Battery(**self.battery_params.__dict__)

        if self.tcl_parameters is not None:
            self.components["tcl"] = TCL(**self.tcl_parameters.__dict__)

        if self.include_infos:
            self.components["information"] = Information(episode=episode,
                                                         episode_length=self.episode_length,
                                                         normalise=self.normalize_observation)

        observation = self._construct_observation()
        return observation, {}

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:

        self.components["intensity"].step()
        self.components["generation"].step()
        self.components["consumption"].step()

        if self.adaptive_consumption_params is not None:
            self.components["adaptive_consumption"].step(action[self.action_slices["adaptive_consumption"]])

        if self.battery_params is not None:
            self.components["battery"].step(action[self.action_slices["battery"]])

        if self.tcl_parameters is not None:
            self.components["tcl"].step(action[self.action_slices["tcl"]],
                                        self.components["information"].state["T2m"].values[0])

        if self.include_infos:
            self.components["information"].step()

        observation = self._construct_observation()
        reward, reward_cache = self._calculate_reward()
        terminated = self._calculate_done()
        truncated = False

        info = {"observation": observation, "action": action, "reward": reward, "reward_cache": reward_cache}

        return observation, reward, terminated, truncated, info

    def _construct_observation(self):
        observation = []

        observation += [self.components["intensity"].state]
        observation += [self.components["generation"].state]
        observation += [self.components["consumption"].state]

        if self.adaptive_consumption_params is not None:
            observation += list(self.components["adaptive_consumption"].state)

        if self.battery_params is not None:
            observation += [self.components["battery"].state]

        if self.tcl_parameters is not None:
            observation += [self.components["tcl"].state]

        if self.include_infos:
            observation += list(self.components["information"].state)
        return np.array(observation, dtype=np.float32)

    def _calculate_reward(self):
        produced_energy = self.components["generation"].reward_cache["G_t"]
        consumed_energy = self.components["consumption"].reward_cache["L_t"]
        if self.adaptive_consumption_params is not None:
            consumed_energy += self.components["adaptive_consumption"].reward_cache["L_{t,ac}"]

        if self.battery_params is not None:
            produced_energy += self.components["battery"].reward_cache["D_t"]
            consumed_energy += self.components["battery"].reward_cache["C_t"]

        if self.tcl_parameters is not None:
            consumed_energy += self.components["tcl"].reward_cache["L_{t,tcl}"] * self.components["tcl"].reward_cache[
                "a_{tcl,t}"]

        reward = self.components["intensity"].reward_cache["I_t"] * (produced_energy - consumed_energy)
        reward_cache = {
            "I_t": self.components["intensity"].reward_cache["I_t"],
            "PE_t": produced_energy,
            "CE_t": consumed_energy,
            "G_t": self.components["generation"].reward_cache["G_t"],
            "L_t": self.components["consumption"].reward_cache["L_t"],
        }
        if self.adaptive_consumption_params is not None:
            reward_cache["L_{t,ac}"] = self.components["adaptive_consumption"].reward_cache["L_{t,ac}"]
        if self.battery_params is not None:
            reward_cache["D_t"] = self.components["battery"].reward_cache["D_t"]
            reward_cache["C_t"] = self.components["battery"].reward_cache["C_t"]
        if self.tcl_parameters is not None:
            reward_cache["L_{t,tcl}"] = self.components["tcl"].reward_cache["L_{t,tcl}"]
            reward_cache["a_{tcl,t}"] = self.components["tcl"].reward_cache["a_{tcl,t}"]
        return reward, reward_cache

    def _calculate_done(self):
        return self.components["intensity"].time >= self.components["intensity"].episode.index[-1]
