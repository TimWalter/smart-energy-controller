from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType

from src.adaptive_consumption import AdaptiveConsumption, AdaptiveConsumptionParameters
from src.battery import Battery, BatteryParameters
from src.consumption import Consumption
from src.electricity_grid import ElectricityGrid
from src.generation import Generation
from src.information import Information
from src.tcl import TCL, TCLParameters


def register(adaptive_consumption_params: AdaptiveConsumptionParameters = AdaptiveConsumptionParameters(),
             battery_params: BatteryParameters = BatteryParameters(),
             tcl_parameters: TCLParameters = TCLParameters(), ):
    kwargs = {
        "adaptive_consumption_params": adaptive_consumption_params,
        "battery_params": battery_params,
        "tcl_parameters": tcl_parameters,
    }

    gym.register(
        id="SingleFamilyHome-v0",
        entry_point="src.environment.environment:SingleFamilyHome",
        max_episode_steps=10080,
        order_enforce=True,
        kwargs=kwargs,
    )


class SingleFamilyHome(gym.Env):
    def __init__(self, adaptive_consumption_params: AdaptiveConsumptionParameters = AdaptiveConsumptionParameters(),
                 battery_params: BatteryParameters = BatteryParameters(),
                 tcl_parameters: TCLParameters = TCLParameters(), ):

        self.adaptive_consumption_params = adaptive_consumption_params
        self.battery_params = battery_params
        self.tcl_parameters = tcl_parameters

        action_space_shape = 0
        observation_space = {
            "generation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "consumption": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
            "intensity": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),

            "day_of_year": gym.spaces.Box(low=1, high=366, shape=(1,), dtype=np.int32),
            "hour_of_day": gym.spaces.Box(low=0, high=23, shape=(1,), dtype=np.int32),
            "solar_irradiation": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
            "solar_elevation": gym.spaces.Box(low=0, high=90, shape=(1,)),
            "temperature": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            "wind_speed": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
        }

        if self.adaptive_consumption_params is not None:
            self.ac_dim = 2 * adaptive_consumption_params.planning_horizon + 1
            self.ac_slice = slice(action_space_shape, action_space_shape + self.ac_dim)
            action_space_shape += self.ac_dim
            observation_space["adaptive_consumption_schedule"] = gym.spaces.Box(low=0, high=np.inf,
                                                                                shape=(self.ac_dim,))

        if self.battery_params is not None:
            self.battery_slice = slice(action_space_shape, action_space_shape + 1)
            action_space_shape += 1
            observation_space["battery_state_of_charge"] = gym.spaces.Box(low=0, high=1, shape=(1,))

        if self.tcl_parameters is not None:
            self.tcl_slice = slice(action_space_shape, action_space_shape + 1)
            action_space_shape += 1
            observation_space["tcl_state_of_charge"] = gym.spaces.Box(low=0, high=1, shape=(1,))

        self.render_mode = "console"
        self.observation_space = gym.spaces.Dict(observation_space)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_space_shape,), dtype=np.float32)

        self.adaptive_consumption = None
        self.battery = None
        self.consumption = None
        self.electricity_grid = None
        self.generation = None
        self.information = None
        self.tcl = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        episode = 0

        if self.adaptive_consumption_params is not None:
            self.adaptive_consumption = AdaptiveConsumption(**self.adaptive_consumption_params.__dict__,
                                                            episode=episode)

        if self.battery_params is not None:
            self.battery = Battery(**self.battery_params.__dict__)

        if self.tcl_parameters is not None:
            self.tcl = TCL(**self.tcl_parameters.__dict__)

        self.consumption = Consumption(episode=episode)

        self.electricity_grid = ElectricityGrid(episode=episode)

        self.generation = Generation(episode=episode)

        self.information = Information(episode=episode)

        observation = self._construct_observation()
        return observation, {}

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        if self.adaptive_consumption_params is not None:
            self.adaptive_consumption.step(action[self.ac_slice])

        if self.battery_params is not None:
            self.battery.step(action[self.battery_slice])

        if self.tcl_parameters is not None:
            self.tcl.step(action[self.tcl_slice], self.information.state["T2m"].values[0])

        self.consumption.step()
        self.electricity_grid.step()
        self.generation.step()
        self.information.step()

        observation = self._construct_observation()
        reward = self._calculate_reward()
        terminated = self._calculate_done()
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _construct_observation(self):
        observation = {
            "generation": self.generation.state,
            "consumption": self.consumption.state,
            "intensity": self.electricity_grid.state,

            "day_of_year": self.information.state.index.dayofyear.values,
            "hour_of_day": self.information.state.index.hour.values,
            "solar_irradiation": self.information.state["G(i)"].values.astype(np.float32),
            "solar_elevation": self.information.state["H_sun"].values.astype(np.float32),
            "temperature": self.information.state["T2m"].values.astype(np.float32),
            "wind_speed": self.information.state["WS10m"].values.astype(np.float32),
        }
        if self.adaptive_consumption_params is not None:
            observation["adaptive_consumption_schedule"] = self.adaptive_consumption.state

        if self.battery_params is not None:
            observation["battery_state_of_charge"] = self.battery.state

        if self.tcl_parameters is not None:
            observation["tcl_state_of_charge"] = self.tcl.state
        return observation

    def _calculate_reward(self):
        produced_energy = self.generation.reward_cache["G_t"]
        consumed_energy = self.consumption.reward_cache["L_t"]
        if self.adaptive_consumption_params is not None:
            consumed_energy += self.adaptive_consumption.reward_cache["s_{a,t}"]

        if self.battery_params is not None:
            produced_energy += self.battery.reward_cache["D_t"]
            consumed_energy += self.battery.reward_cache["C_t"]

        if self.tcl_parameters is not None:
            consumed_energy += self.tcl.reward_cache["L_{TCL}"] * self.tcl.reward_cache["a_{tcl,t}"]

        reward = self.electricity_grid.reward_cache["I_t"] * (produced_energy - consumed_energy)
        return float(reward[0])

    def _calculate_done(self):
        return self.information.time >= self.information.episode.index[-1]
