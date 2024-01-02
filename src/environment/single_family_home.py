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
    ees: ExternalElectricitySupply
    hed: HouseholdEnergyDemand
    rsa: RooftopSolarArray
    ess: EnergyStorageSystem
    fdr: FlexibleDemandResponse
    tcl: ThermostaticallyControlledLoad
    wtd: WeatherAndTimeData

    def __init__(self):
        try:
            self.config = json.load(open("environment/config.json", "r"))
        except FileNotFoundError:
            self.config = json.load(open("config.json", "r"))

        self.ess_condition = "energy_storage_system" in self.config.keys()
        self.fdr_condition = "flexible_demand_response" in self.config.keys()
        self.tcl_condition = "thermostatically_controlled_load" in self.config.keys()
        self.wtd_condition = self.config["include_additional_information"]

        self.next_episode = 0
        self.episode_set = [0]

        self.develop_set = [0]
        self.test_set = [25, 50, 75, 100]
        self.train_set = np.random.choice(np.setdiff1d(np.arange(0, 104), self.test_set), 95, replace=False)
        self.eval_set = np.setdiff1d(np.setdiff1d(np.arange(0, 104), self.test_set), self.train_set)

        self.shuffle = False
        if self.ess_condition:
            self.energy_storage_system_upper_bound = self.config["energy_storage_system"]["capacity_in_kwmin"]
        if self.tcl_condition:
            self.tcl_lower_bound = self.config["thermostatically_controlled_load"]["minimal_temperature"]
            self.tcl_upper_bound = self.config["thermostatically_controlled_load"]["maximal_temperature"]
        self.shuffled_initial_conditions = [
            np.random.uniform(0, self.energy_storage_system_upper_bound * 0.2, 95)
            if self.ess_condition else None,
            np.random.uniform(0, (self.tcl_upper_bound - self.tcl_lower_bound) * 0.2 + self.tcl_lower_bound, 95)
            if self.tcl_condition else None
        ]

        self.observation_space = self._observation_space()
        self.action_space = self._action_space()
        self.action_slice = self._action_slice()

    def develop(self):
        self.next_episode = 0
        self.episode_set = self.develop_set
        self.shuffle = False

    def train(self):
        self.next_episode = 0
        self.episode_set = self.train_set
        self.shuffle = True

    def eval(self):
        self.next_episode = 0
        self.episode_set = self.eval_set
        self.shuffle = True

    def test(self):
        self.next_episode = 0
        self.episode_set = self.test_set
        self.shuffle = True

    def _observation_space(self) -> gym.spaces.Dict:
        spaces = {
            "carbon_intensity": gym.spaces.Box(low=0.0511, high=2.3729),
            "household_energy_demand": gym.spaces.Box(low=0.0, high=10.1619),
            "rooftop_solar_generation": gym.spaces.Box(low=-0.4269, high=44.0072)
        }

        if self.ess_condition:
            spaces["energy_storage_system_charge"] = gym.spaces.Box(low=0.0, high=self.config["energy_storage_system"][
                "capacity_in_kwmin"])

        if self.fdr_condition:
            dim = self.config["flexible_demand_response"]["planning_horizon"] * 2 + 1
            spaces["flexible_demand_schedule"] = gym.spaces.Box(low=0.0, high=8.5839 * dim)

        if self.tcl_condition:
            spaces["tcl_indoor_temperature"] = gym.spaces.Box(
                low=self.tcl_lower_bound,
                high=self.tcl_upper_bound)

        if self.wtd_condition:
            spaces["day_of_year"] = gym.spaces.Box(low=1, high=365)
            spaces["hour_of_day"] = gym.spaces.Box(low=0, high=23)
            spaces["solar_irradiation"] = gym.spaces.Box(low=0.0, high=1082.1)
            spaces["solar_elevation"] = gym.spaces.Box(low=0.0, high=64.41)
            spaces["temperature"] = gym.spaces.Box(low=-10.43, high=35.13)
            spaces["wind_speed"] = gym.spaces.Box(low=0.0, high=12.76)

        return gym.spaces.Dict(spaces, seed=42)

    def _construct_observation(self) -> ObsType:
        observation = {
            "carbon_intensity": self.ees.state,
            "household_energy_demand": self.hed.state,
            "rooftop_solar_generation": self.rsa.state
        }

        if self.ess_condition:
            observation["energy_storage_system_charge"] = self.ess.state

        if self.fdr_condition:
            observation["flexible_demand_schedule"] = np.sum(self.fdr.state)

        if self.tcl_condition:
            observation["tcl_indoor_temperature"] = self.tcl.state

        if self.wtd_condition:
            keys = ["day_of_year", "hour_of_day", "solar_irradiation", "solar_elevation", "temperature", "wind_speed"]
            for i, key in enumerate(keys):
                observation[key] = self.wtd.state[i]

        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                observation[key] = value.astype(np.float32)
            else:
                observation[key] = np.array([value], dtype=np.float32)

        return observation

    def _action_space(self) -> gym.spaces.MultiDiscrete | gym.spaces.Box:
        if self.config["action_space"]["type"] == "discrete":
            nvec = []
            if self.ess_condition:
                nvec += [self.config["action_space"]["levels"][0]]
            if self.fdr_condition:
                nvec += [self.config["action_space"]["levels"][1 if self.ess_condition else 0]]
            if self.tcl_condition:
                nvec += [self.config["action_space"]["levels"][-1]]
            return gym.spaces.MultiDiscrete(nvec, seed=42)
        elif self.config["action_space"]["type"] == "continuous":
            low = []
            high = []
            if self.ess_condition:
                low += [-1]
                high += [1]
            if self.fdr_condition:
                low += [-1]
                high += [1]
            if self.tcl_condition:
                low += [-1]
                high += [1]
            return gym.spaces.Box(low=np.array(low), high=np.array(high), seed=42)
        else:
            raise ValueError("Invalid action space type.")

    def _action_slice(self) -> dict[str, slice | int]:
        action_slice = {}
        if self.ess_condition:
            action_slice["energy_storage_system"] = 0
        if self.fdr_condition:
            action_slice["flexible_demand_response"] = 1 if self.ess_condition else 0
        if self.tcl_condition:
            action_slice["thermostatically_controlled_load"] = -1
        return action_slice

    def _rescale_discrete_action(self, action: ActType) -> ActType:
        rescaled_action = []

        if self.ess_condition:
            energy_storage_system_action = action[self.action_slice["energy_storage_system"]]
            levels = self.action_space.nvec[self.action_slice["energy_storage_system"]]
            rescaled_action += [2 * energy_storage_system_action / (levels - 1) - 1]  # [0, levels-1] -> [-1, 1]
        if self.fdr_condition:
            flexible_demand_response_action = action[self.action_slice["flexible_demand_response"]]
            levels = self.action_space.nvec[self.action_slice["flexible_demand_response"]]
            rescaled_action += [2 * flexible_demand_response_action / (levels - 1) - 1]  # [0, levels-1] -> [-1, 1]
        if self.tcl_condition:
            thermostatically_controlled_load_action = action[self.action_slice["thermostatically_controlled_load"]]
            levels = self.action_space.nvec[self.action_slice["thermostatically_controlled_load"]]
            rescaled_action += [thermostatically_controlled_load_action / (levels - 1)]  # [0, levels-1] -> [0, 1]

        return np.array(rescaled_action, dtype=np.float32).flatten()

    def _rescale_continuous_action(self, action: ActType) -> ActType:
        rescaled_action = []

        if self.ess_condition:
            energy_storage_system_action = action[self.action_slice["energy_storage_system"]]
            rescaled_action += [energy_storage_system_action]  # [-1, 1] -> [-1, 1]
        if self.fdr_condition:
            flexible_demand_response_action = action[self.action_slice["flexible_demand_response"]]
            rescaled_action += [flexible_demand_response_action]  # [-1, 1] -> [-1, 1]
        if self.tcl_condition:
            thermostatically_controlled_load_action = action[self.action_slice["thermostatically_controlled_load"]]
            rescaled_action += [(thermostatically_controlled_load_action + 1) / 2]  # [-1, 1] -> [0, 1]

        return np.array(rescaled_action, dtype=np.float32).flatten()

    def _calculate_reward(self) -> float:
        produced_energy = self.rsa.reward_cache["rooftop_solar_generation"]
        consumed_energy = self.hed.reward_cache["household_energy_demand"]

        if self.ess_condition:
            consumed_energy += self.ess.reward_cache["consumed_energy"]

        if self.fdr_condition:
            consumed_energy += self.fdr.reward_cache["consumed_and_discounted_energy"]

        if self.tcl_condition:
            consumed_energy += self.tcl.reward_cache["consumed_energy"]
            discomfort = self.tcl.penalty_factor * (self.tcl.reward_cache["indoor_temperature"] - (
                        self.tcl.maximal_temperature + self.tcl.minimal_temperature) / 2) ** 2

        reward = self.ees.reward_cache["carbon_intensity"] * (produced_energy - consumed_energy)
        if self.tcl_condition:
            reward -= discomfort
        return reward

    def _calculate_done(self) -> bool:
        return self.ees.time >= self.ees.episode.index[-1]

    def reset(self, seed: int | None = 42, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        episode = self.episode_set[self.next_episode]
        self.next_episode = (self.next_episode + 1) % len(self.episode_set)

        self.ees = ExternalElectricitySupply(episode=episode)
        self.hed = HouseholdEnergyDemand(episode=episode)
        self.rsa = RooftopSolarArray(episode=episode)

        if self.ess_condition:
            self.ess = EnergyStorageSystem(**self.config["energy_storage_system"])
            if self.shuffle:
                self.ess.charge = self.shuffled_initial_conditions[0][episode]
                self.ess.update_state()

        if self.fdr_condition:
            self.fdr = FlexibleDemandResponse(**self.config["flexible_demand_response"],
                                              episode=episode)

        if self.tcl_condition:
            self.tcl = ThermostaticallyControlledLoad(
                **self.config["thermostatically_controlled_load"])
            if self.shuffle:
                self.tcl.indoor_temperature = self.shuffled_initial_conditions[1][episode]
                self.tcl.building_mass_temperature = self.shuffled_initial_conditions[1][
                    episode]
                self.tcl.update_state()

        if self.wtd_condition or self.tcl_condition:
            self.wtd = WeatherAndTimeData(episode=episode)

        observation = self._construct_observation()
        return observation, {}

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict]:
        if self.config["action_space"]["type"] == "discrete":
            rescaled_action = self._rescale_discrete_action(action)
        else:
            rescaled_action = self._rescale_continuous_action(action)
        self.ees.step()
        self.hed.step()
        self.rsa.step()

        if self.ess_condition:
            self.ess.step(rescaled_action[self.action_slice["energy_storage_system"]])

        if self.fdr_condition:
            self.fdr.step(rescaled_action[self.action_slice["flexible_demand_response"]])

        if self.tcl_condition:
            self.tcl.step(
                rescaled_action[self.action_slice["thermostatically_controlled_load"]],
                self.wtd.state[4])

        if self.wtd_condition or self.tcl_condition:
            self.wtd.step()

        observation = self._construct_observation()
        reward = self._calculate_reward()
        terminated = self._calculate_done()
        truncated = False

        info = {"next_observation": observation, "action": rescaled_action, "reward": reward,
                "cache": {
                    "given_reward": self.ees.reward_cache["carbon_intensity"] * (
                            self.rsa.reward_cache["rooftop_solar_generation"] -
                            self.hed.reward_cache["household_energy_demand"]),
                }}
        if self.ess_condition:
            info["cache"]["battery_reward"] = self.ees.reward_cache["carbon_intensity"] * (
                -self.ess.reward_cache["consumed_energy"])
        if self.fdr_condition:
            info["cache"]["fdr_reward"] = self.ees.reward_cache["carbon_intensity"] * (
                -self.fdr.reward_cache["consumed_and_discounted_energy"])
        if self.tcl_condition:
            info["cache"]["tcl_reward"] = self.ees.reward_cache["carbon_intensity"] * (
                -self.tcl.reward_cache["consumed_energy"])
            info["cache"]["discomfort"] = -self.tcl.penalty_factor * (self.tcl.reward_cache["indoor_temperature"] - (
                                                      self.tcl.maximal_temperature + self.tcl.minimal_temperature) / 2) ** 2

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(SingleFamilyHome())
    check_env(env)

    observation = env.reset()[0]
    print(f"[RESET] battery_state_of_charge: {observation['battery_state_of_charge']}",
          f" flexible_demand_schedule: {observation['flexible_demand_schedule']}",
          f" tcl_state_of_charge: {observation['tcl_state_of_charge']}")

    if env.unwrapped.config["action_space"]["type"] == "discrete":
        actions = [np.zeros(3)] * 20

        charge_action = np.zeros(3)
        charge_action[0] = 100
        actions += [charge_action]
        discharge_action = np.zeros(3)
        discharge_action[0] = 0
        actions += [discharge_action]

        delay_action = np.zeros(3)
        delay_action[1] = 0
        actions += [delay_action]
        expedite_action = np.zeros(3)
        expedite_action[1] = 100
        actions += [expedite_action]

        heating_action = np.zeros(3)
        heating_action[-1] = 100
        actions += [heating_action]

    elif env.unwrapped.config["action_space"]["type"] == "continuous":
        actions = [-1 * np.ones(3)] * 20

        charge_action = -1 * np.ones(1442)
        charge_action[0] = 1
        actions += [charge_action]
        discharge_action = -1 * np.ones(3)
        discharge_action[0] = -1
        actions += [discharge_action]

        delay_action = -1 * np.ones(3)
        delay_action[1] = -1
        actions += [delay_action]
        expedite_action = -1 * np.ones(3)
        expedite_action[1] = 1
        actions += [expedite_action]

        heating_action = -1 * np.ones(3)
        heating_action[-1] = 1
        actions += [heating_action]

    for i, action in enumerate(actions):
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"[{i}] battery_state_of_charge: {observation['battery_state_of_charge']}"
              f" flexible_demand_schedule: {observation['flexible_demand_schedule']}"
              f" tcl_state_of_charge: {observation['tcl_state_of_charge']}"
              f", Reward: {reward} Action: {info['action']}")
