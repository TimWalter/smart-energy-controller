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
    metadata = {"render_modes": ["console"]}

    def __init__(self, adaptive_consumption_params: AdaptiveConsumptionParameters = AdaptiveConsumptionParameters(),
                 battery_params: BatteryParameters = BatteryParameters(),
                 tcl_parameters: TCLParameters = TCLParameters(), ):

        self.adaptive_consumption_params = adaptive_consumption_params
        self.battery_params = battery_params
        self.tcl_parameters = tcl_parameters

        self.ac_dim = 2 * adaptive_consumption_params.planning_horizon + 1

        self.render_mode = "console"
        self.observation_space = gym.spaces.Dict(
            {
                "battery_state_of_charge": gym.spaces.Box(low=0, high=1, shape=(1,)),
                "tcl_state_of_charge": gym.spaces.Box(low=0, high=1, shape=(1,)),
                "adaptive_consumption_schedule": gym.spaces.Box(low=0, high=np.inf, shape=(self.ac_dim,)),

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
        )

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.battery_slice = slice(0, 1)
        self.tcl_slice = slice(1, 2)

        #self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.ac_dim + 2,), dtype=np.float32)
        #self.ac_slice = slice(0, self.ac_dim)
        #self.battery_slice = slice(self.ac_dim, self.ac_dim + 1)
        #self.tcl_slice = slice(self.ac_dim + 1, self.ac_dim + 2)

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

        self.adaptive_consumption = AdaptiveConsumption(**self.adaptive_consumption_params.__dict__, episode=episode)

        self.battery = Battery(**self.battery_params.__dict__)

        self.consumption = Consumption(episode=episode)

        self.electricity_grid = ElectricityGrid(episode=episode)

        self.generation = Generation(episode=episode)

        self.information = Information(episode=episode)

        self.tcl = TCL(**self.tcl_parameters.__dict__)

        observation = self._construct_observation()
        return observation, {}

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        #self.adaptive_consumption.step(np.ones(self.ac_dim))
        #self.adaptive_consumption.step(action[self.ac_slice])
        self.battery.step(action[self.battery_slice])
        self.consumption.step()
        self.electricity_grid.step()
        self.generation.step()

        self.tcl.step(action[self.tcl_slice], self.information.state["T2m"].values[0])
        self.information.step()

        observation = self._construct_observation()
        reward = self._calculate_reward()
        terminated = self._calculate_done()
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _construct_observation(self):
        return {
            "battery_state_of_charge": self.battery.state,
            "tcl_state_of_charge": self.tcl.state,
            "adaptive_consumption_schedule": self.adaptive_consumption.state,

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

    def _calculate_reward(self):
        produced_energy = self.generation.reward_cache["G_t"] + self.battery.reward_cache["D_t"]
        #consumed_energy = (self.consumption.reward_cache["L_t"] +
        #                   self.battery.reward_cache["C_t"] +
        #                   self.tcl.reward_cache["L_{TCL}"] * self.tcl.reward_cache["a_{tcl,t}"] +
        #                   self.adaptive_consumption.reward_cache["s_{a,t}"])

        consumed_energy = (self.consumption.reward_cache["L_t"] +
                           self.battery.reward_cache["C_t"] +
                           self.tcl.reward_cache["L_{TCL}"] * self.tcl.reward_cache["a_{tcl,t}"])

        reward = self.electricity_grid.reward_cache["I_t"] * (produced_energy - consumed_energy)
        return float(reward[0])

    def _calculate_done(self):
        return self.information.time >= self.information.episode.index[-1]
