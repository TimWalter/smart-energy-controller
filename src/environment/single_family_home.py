import json
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType

from environment.components.energy_storage_system import EnergyStorageSystem
from environment.components.external_electricity_supply import ExternalElectricitySupply
from environment.components.flexible_demand_response import FlexibleDemandResponse
from environment.components.household_energy_demand import HouseholdEnergyDemand
from environment.components.rooftop_solar_array import RooftopSolarArray
from environment.components.thermostatically_controlled_load import ThermostaticallyControlledLoad
from environment.components.weather_and_time_data import WeatherAndTimeData


class SingleFamilyHome(gym.Env):
    external_electricity_supply: ExternalElectricitySupply
    household_energy_demand: HouseholdEnergyDemand
    rooftop_solar_array: RooftopSolarArray
    energy_storage_system: EnergyStorageSystem
    flexible_demand_response: FlexibleDemandResponse
    thermostatically_controlled_load: ThermostaticallyControlledLoad
    weather_and_time_data: WeatherAndTimeData

    def __init__(self):
        try:
            self.config = json.load(open("environment/config.json", "r"))
        except FileNotFoundError:
            self.config = json.load(open("config.json", "r"))
        self.energy_storage_system_condition = "energy_storage_system" in self.config.keys()
        self.flexible_demand_response_condition = "flexible_demand_response" in self.config.keys()
        self.thermostatically_controlled_load_condition = "thermostatically_controlled_load" in self.config.keys()
        self.additional_information_condition = "include_additional_information" in self.config.keys()

        self.observation_space = self._observation_space()
        self.action_space = self._action_space()
        self.action_slice = self._action_slice()

    def _observation_space(self) -> gym.spaces.Dict:
        spaces = {
            "carbon_intensity": gym.spaces.Box(low=0.0511, high=2.3729),
            "household_energy_demand": gym.spaces.Box(low=0.0, high=10.1619),
            "rooftop_solar_generation": gym.spaces.Box(low=-0.4269, high=44.0072)
        }

        if self.energy_storage_system_condition:
            spaces["battery_state_of_charge"] = gym.spaces.Box(low=0.0, high=1.0)

        if self.flexible_demand_response_condition:
            dim = self.config["flexible_demand_response"]["planning_horizon"] * 2 + 1
            spaces["flexible_demand_schedule"] = gym.spaces.Box(low=0.0, high=8.5839, shape=(dim,))

        if self.thermostatically_controlled_load_condition:
            spaces["tcl_state_of_charge"] = gym.spaces.Box(low=0.0, high=1.0)

        if self.additional_information_condition:
            spaces["day_of_year"] = gym.spaces.Box(low=1, high=365)
            spaces["hour_of_day"] = gym.spaces.Box(low=0, high=23)
            spaces["solar_irradiation"] = gym.spaces.Box(low=0.0, high=1082.1)
            spaces["solar_elevation"] = gym.spaces.Box(low=0.0, high=64.41)
            spaces["temperature"] = gym.spaces.Box(low=-10.43, high=35.13)
            spaces["wind_speed"] = gym.spaces.Box(low=0.0, high=12.76)

        return gym.spaces.Dict(spaces, seed=42)

    def _construct_observation(self) -> ObsType:
        observation = {
            "carbon_intensity": self.external_electricity_supply.state,
            "household_energy_demand": self.household_energy_demand.state,
            "rooftop_solar_generation": self.rooftop_solar_array.state
        }

        if self.energy_storage_system_condition:
            observation["battery_state_of_charge"] = self.energy_storage_system.state

        if self.flexible_demand_response_condition:
            observation["flexible_demand_schedule"] = self.flexible_demand_response.state

        if self.thermostatically_controlled_load_condition:
            observation["tcl_state_of_charge"] = self.thermostatically_controlled_load.state

        if self.additional_information_condition:
            keys = ["day_of_year", "hour_of_day", "solar_irradiation", "solar_elevation", "temperature", "wind_speed"]
            for i, key in enumerate(keys):
                observation[key] = self.weather_and_time_data.state[i]

        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                observation[key] = value.astype(np.float32)
            else:
                observation[key] = np.array([value], dtype=np.float32)

        return observation

    def _action_space(self) -> gym.spaces.MultiDiscrete | gym.spaces.Box:
        if self.config["action_space"]["type"] == "discrete":
            nvec = []
            if self.energy_storage_system_condition:
                nvec += [self.config["action_space"]["levels"][0]]
            if self.flexible_demand_response_condition:
                nvec += [self.config["action_space"]["levels"][1 if self.energy_storage_system_condition else 0]] * \
                        (2 * self.config["flexible_demand_response"]["planning_horizon"] + 1)
            if self.thermostatically_controlled_load_condition:
                nvec += [self.config["action_space"]["levels"][-1]]
            return gym.spaces.MultiDiscrete(nvec, seed=42)
        elif self.config["action_space"]["type"] == "continuous":
            low = []
            high = []
            if self.energy_storage_system_condition:
                low += [-1]
                high += [1]
            if self.flexible_demand_response_condition:
                low += [-1] * (2 * self.config["flexible_demand_response"]["planning_horizon"] + 1)
                high += [1] * (2 * self.config["flexible_demand_response"]["planning_horizon"] + 1)
            if self.thermostatically_controlled_load_condition:
                low += [-1]
                high += [1]
            return gym.spaces.Box(low=np.array(low), high=np.array(high), seed=42)
        else:
            raise ValueError("Invalid action space type.")

    def _action_slice(self) -> dict[str, slice]:
        action_slice = {}
        if self.energy_storage_system_condition:
            action_slice["energy_storage_system"] = 0
        if self.flexible_demand_response_condition:
            start = 1 if self.energy_storage_system_condition else 0
            stop = start + 2 * self.config["flexible_demand_response"]["planning_horizon"] + 1
            action_slice["flexible_demand_response"] = slice(start, stop)
        if self.thermostatically_controlled_load_condition:
            action_slice["thermostatically_controlled_load"] = -1
        return action_slice

    def _rescale_discrete_action(self, action: ActType) -> ActType:
        rescaled_action = []

        if self.energy_storage_system_condition:
            energy_storage_system_action = action[self.action_slice["energy_storage_system"]]
            levels = self.action_space.nvec[self.action_slice["energy_storage_system"]]
            rescaled_action += [2 * energy_storage_system_action / (levels - 1) - 1]  # [0, levels-1] -> [-1, 1]
        if self.flexible_demand_response_condition:
            flexible_demand_response_action = action[self.action_slice["flexible_demand_response"]]
            levels = self.action_space.nvec[self.action_slice["flexible_demand_response"]]
            rescaled_action += list(flexible_demand_response_action / (levels - 1))  # [0, levels-1] -> [0, 1]
        if self.thermostatically_controlled_load_condition:
            thermostatically_controlled_load_action = action[self.action_slice["thermostatically_controlled_load"]]
            levels = self.action_space.nvec[self.action_slice["thermostatically_controlled_load"]]
            rescaled_action += [thermostatically_controlled_load_action / (levels - 1)]  # [0, levels-1] -> [0, 1]

        return np.array(rescaled_action, dtype=np.float32)

    def _rescale_continuous_action(self, action: ActType) -> ActType:
        rescaled_action = []

        if self.energy_storage_system_condition:
            energy_storage_system_action = action[self.action_slice["energy_storage_system"]]
            rescaled_action += [energy_storage_system_action]  # [-1, 1] -> [-1, 1]
        if self.flexible_demand_response_condition:
            flexible_demand_response_action = action[self.action_slice["flexible_demand_response"]]
            rescaled_action += list((flexible_demand_response_action + 1) / 2)  # [-1, 1] -> [0, 1]
        if self.thermostatically_controlled_load_condition:
            thermostatically_controlled_load_action = action[self.action_slice["thermostatically_controlled_load"]]
            rescaled_action += [(thermostatically_controlled_load_action + 1) / 2]  # [-1, 1] -> [0, 1]

        return np.array(rescaled_action, dtype=np.float32)

    def _calculate_reward(self) -> float:
        produced_energy = self.rooftop_solar_array.reward_cache["rooftop_solar_generation"]
        consumed_energy = self.household_energy_demand.reward_cache["household_energy_demand"]
        if self.energy_storage_system_condition:
            produced_energy += self.energy_storage_system.reward_cache["discharge_rate"]
            consumed_energy += self.energy_storage_system.reward_cache["charge_rate"]

        if self.flexible_demand_response_condition:
            consumed_energy += self.flexible_demand_response.reward_cache["flexible_demand_response"]

        if self.thermostatically_controlled_load_condition:
            consumed_energy += self.thermostatically_controlled_load.reward_cache["nominal_power"] * \
                               self.thermostatically_controlled_load.reward_cache["tcl_action"]

        reward = self.external_electricity_supply.reward_cache["carbon_intensity"] * (produced_energy - consumed_energy)
        return reward

    def _calculate_done(self) -> bool:
        return self.external_electricity_supply.time >= self.external_electricity_supply.episode.index[-1]

    def reset(self, seed: int | None = 42, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        episode = 0

        self.external_electricity_supply = ExternalElectricitySupply(episode=episode)
        self.household_energy_demand = HouseholdEnergyDemand(episode=episode)
        self.rooftop_solar_array = RooftopSolarArray(episode=episode)

        if self.energy_storage_system_condition:
            self.energy_storage_system = EnergyStorageSystem(**self.config["energy_storage_system"])

        if self.flexible_demand_response_condition:
            self.flexible_demand_response = FlexibleDemandResponse(**self.config["flexible_demand_response"],
                                                                   episode=episode)

        if self.thermostatically_controlled_load_condition:
            self.thermostatically_controlled_load = ThermostaticallyControlledLoad(
                **self.config["thermostatically_controlled_load"])

        if self.additional_information_condition or self.thermostatically_controlled_load_condition:
            self.weather_and_time_data = WeatherAndTimeData(episode=episode)

        observation = self._construct_observation()
        return observation, {}

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        if self.config["action_space"]["type"] == "discrete":
            rescaled_action = self._rescale_discrete_action(action)
        else:
            rescaled_action = self._rescale_continuous_action(action)
        self.external_electricity_supply.step()
        self.household_energy_demand.step()
        self.rooftop_solar_array.step()

        if self.energy_storage_system_condition:
            self.energy_storage_system.step(rescaled_action[self.action_slice["energy_storage_system"]])

        if self.flexible_demand_response_condition:
            self.flexible_demand_response.step(rescaled_action[self.action_slice["flexible_demand_response"]])

        if self.thermostatically_controlled_load_condition:
            self.thermostatically_controlled_load.step(
                rescaled_action[self.action_slice["thermostatically_controlled_load"]],
                self.weather_and_time_data.state[4])

        if self.additional_information_condition or self.thermostatically_controlled_load_condition:
            self.weather_and_time_data.step()

        observation = self._construct_observation()
        reward = self._calculate_reward()
        terminated = self._calculate_done()
        truncated = False

        info = {"next_observation": observation, "action": rescaled_action, "reward": reward}

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(SingleFamilyHome())
    check_env(env)

    observation = env.reset()
    print(f"Observation: {observation}")

    if env.unwrapped.config["action_space"]["type"] == "discrete":
        actions = [np.zeros(1442)] * 20

        charge_action = np.zeros(1442)
        charge_action[0] = 4
        actions += [charge_action]
        discharge_action = np.zeros(1442)
        discharge_action[0] = 0
        actions += [discharge_action]

        delay_action = np.zeros(1442)
        delay_action[721] = 1
        actions += [delay_action]
        expedite_action = np.zeros(1442)
        expedite_action[722] = 1
        actions += [expedite_action]

        heating_action = np.zeros(1442)
        heating_action[-1] = 10
        actions += [heating_action]

    elif env.unwrapped.config["action_space"]["type"] == "continuous":
        actions = [-1 * np.ones(1442)] * 20

        charge_action = -1 * np.ones(1442)
        charge_action[0] = 1
        actions += [charge_action]
        discharge_action = -1 * np.ones(1442)
        discharge_action[0] = -1
        actions += [discharge_action]

        delay_action = -1 * np.ones(1442)
        delay_action[721] = 1
        actions += [delay_action]
        expedite_action = -1 * np.ones(1442)
        expedite_action[722] = 1
        actions += [expedite_action]

        heating_action = -1 * np.ones(1442)
        heating_action[-1] = 1
        actions += [heating_action]

    for i, action in enumerate(actions):
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"[{i}] Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, "
              f"Info: {info}")
