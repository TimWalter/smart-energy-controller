from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType

from src.environment.adaptive_consumption import AdaptiveConsumption, AdaptiveConsumptionParameters
from src.environment.battery import Battery, BatteryParameters
from src.environment.consumption import Consumption
from src.environment.electricity_grid import ElectricityGrid
from src.environment.generation import Generation
from src.environment.information import Information
from src.environment.tcl import TCL, TCLParameters


class SingleFamilyHome(gym.Env):
    def __init__(self, adaptive_consumption_params: AdaptiveConsumptionParameters = AdaptiveConsumptionParameters(),
                 battery_params: BatteryParameters = BatteryParameters(),
                 tcl_parameters: TCLParameters = TCLParameters(), ):
        self.observation_space = gym.spaces.Dict(
            {
                "controlled": gym.spaces.Dict(
                    {
                        "battery_state_of_charge": gym.spaces.Box(low=0, high=1, shape=(1,)),
                        "tcl_state_of_charge": gym.spaces.Box(low=0, high=1, shape=(1,)),
                        "adaptive_consumption_schedule": gym.spaces.MultiBinary(
                            2 * adaptive_consumption_params.planning_horizon + 1),
                    }
                ),
                "uncontrolled": gym.spaces.Dict({
                    "generation": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "consumption": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "intensity": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                }),
                "informational": gym.spaces.Dict({
                    "time": gym.spaces.Text(min_length=26, max_length=26, charset="0123456789-:T."),
                    "solar_irradiation": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "solar_elevation": gym.spaces.Box(low=0, high=90, shape=(1,)),
                    "temperature": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                    "wind_speed": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                }),
            }
        )

        self.action_space = gym.spaces.Dict({
            "battery": gym.spaces.Box(low=-1, high=1, shape=(1,)),
            "tcl": gym.spaces.Box(low=0, high=1, shape=(1,)),
            "adaptive_consumption_action": gym.spaces.MultiBinary(2 * adaptive_consumption_params.planning_horizon + 1),
        })

        self.adaptive_consumption_params = adaptive_consumption_params
        self.battery_params = battery_params
        self.tcl_parameters = tcl_parameters

        self.adaptive_consumption = None
        self.battery = None
        self.consumption = None
        self.electricity_grid = None
        self.generation = None
        self.information = None
        self.tcl = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        episode = self.np_random.integers(0, 104)

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
        self.adaptive_consumption.step(action["adaptive_consumption_schedule"])
        self.battery.step(action["battery"])
        self.consumption.step()
        self.electricity_grid.step()
        self.generation.step()

        self.tcl.step(action["tcl"], self.information.state["T2m"])
        self.information.step()

        observation = self._construct_observation()
        reward = self._calculate_reward()
        terminated = self._calculate_done()
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _construct_observation(self):
        return {
            "controlled": {
                "battery_state_of_charge": self.battery.state,
                "tcl_state_of_charge": self.tcl.state,
                "adaptive_consumption_schedule": self.adaptive_consumption.state,
            },
            "uncontrolled": {
                "generation": self.generation.state,
                "consumption": self.consumption.state,
                "intensity": self.electricity_grid.state,
            },
            "informational": {
                "time": self.information.state.index.toisoformat(),
                "solar_irradiation": self.information.state["G(i)"],
                "solar_elevation": self.information.state["H_sun"],
                "temperature": self.information.state["T2m"],
                "wind_speed": self.information.state["WS10m"],
            },
        }

    def _calculate_reward(self):
        produced_energy = self.generation.reward_cache["G_t"] + self.battery.reward_cache["D_t"]
        consumed_energy = (self.consumption.reward_cache["L_t"] +
                           self.battery.reward_cache["C_t"] +
                           self.tcl.reward_cache["L_{TCL}"] * self.tcl.reward_cache["a_{tcl,t}"] +
                           self.adaptive_consumption.reward_cache["s_{a,t}"])

        reward = self.electricity_grid.reward_cache["I_t"] * (produced_energy - consumed_energy)
        return reward

    def _calculate_done(self):
        return self.information.time >= self.information.episode.index[-1]
