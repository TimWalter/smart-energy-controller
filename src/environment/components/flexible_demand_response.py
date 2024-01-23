from datetime import timedelta

import numpy as np
import pandas as pd

from src.environment.components.base.component import Component
from src.environment.components.base.data_loader import DataLoader


class FlexibleDemandResponse(Component, DataLoader):
    """
    Represents a flexible demand response.

    This class inherits from the Component and DataLoader classes.
    """

    def __init__(self, planning_horizon: int, deterministic: bool, patience: float, resolution: str):
        """
        Initializes the FlexibleDemandResponse. All energy expenditures are in kWmin.

        Args:
            planning_horizon (int): The planning horizon.
            deterministic (bool): If the response is deterministic.
            patience (float): The patience factor.
            episode (int): The current episode. Defaults to 0.
        """
        Component.__init__(self)
        DataLoader.__init__(self, file=f'../data/{resolution}/flexible_demand_response.h5', resolution=resolution)

        self.planning_horizon = planning_horizon
        self.deterministic = deterministic
        self.patience = patience
        self.resolution = resolution

        self.planning_timedelta = timedelta(seconds=planning_horizon * self.second_to_resolution)
        self.schedule = np.concatenate([np.ones(self.planning_horizon + 1), np.zeros(self.planning_horizon)])
        if self.deterministic:
            self.patience_weighting = np.ones_like(self.schedule)
        else:
            self.patience_weighting = np.exp(
                -1 / self.patience * np.abs(np.arange(len(self.schedule)) - self.planning_horizon))

        self.consume_weighting = np.concatenate(
            [np.arange(1 / self.planning_horizon, 1 + 1 / self.planning_horizon, 1 / self.planning_horizon),
             np.ones(self.planning_horizon + 1)])
        self.discount_weighting = np.concatenate(
            [np.ones(self.planning_horizon) / self.planning_horizon, np.zeros(self.planning_horizon + 1)])

    def reset(self, episode: int):
        self.set_episode(episode)
        self.update_state()

    def step(self, action: float):
        """
        Perform a step in the environment.

        Args:
            action (np.ndarray): The action is in [-1, 1]. -1 maximum delay, 1 maximum expedite.
        """
        execution_probability = np.clip(self.schedule + action * self.patience_weighting, 0, 1)
        coins = np.random.uniform(0, 1, self.schedule.shape)

        executed_actions = coins < execution_probability

        self.update_reward_cache(executed_actions)

        # Set power to 0 for executed actions and discount the delayed actions
        self.state[executed_actions] = 0
        self.set_values(self.state, self.time - self.planning_timedelta, self.time + self.planning_timedelta, "energy")

        self.step_time()
        self.update_state()

    def update_state(self):
        """
        Update the flexible demand response window.
        """
        self.state = self.get_values(self.time - self.planning_timedelta, self.time + self.planning_timedelta)  # in kW

        rows_to_add = len(self.schedule) - self.state.shape[0]

        # Ensure zero padding
        if rows_to_add > 0:
            if self.time - self.planning_timedelta < self.state.index[0]:
                # If time is before the start of the state DataFrame, prepend the rows
                zero_data = pd.DataFrame({"energy": [0] * rows_to_add},
                                         index=pd.date_range(end=self.state.index[0] - self.one_timestep_delta,
                                                             periods=rows_to_add, freq='min'))
                self.state = pd.concat([zero_data, self.state])
            else:
                # If time is after the end of the state DataFrame, append the rows
                zero_data = pd.DataFrame({"energy": [0] * rows_to_add},
                                         index=pd.date_range(start=self.state.index[-1] + self.one_timestep_delta,
                                                             periods=rows_to_add, freq='min'))
                self.state = pd.concat([self.state, zero_data])

        self.state = self.state["energy"].values

    def update_reward_cache(self, executed_actions: np.ndarray) -> np.ndarray:
        """
        Update the reward cache with the power.

        Args:
            executed_actions (ndarray): The executed actions.
        """
        executed_mask = np.zeros_like(self.state, dtype=bool)
        executed_mask[executed_actions] = 1

        consumed_energy = self.state * executed_mask
        delayed_energy = self.state * (~executed_mask)
        consumed_and_discounted_energy = consumed_energy * self.consume_weighting + delayed_energy * self.discount_weighting

        self.reward_cache["consumed_and_discounted_energy"] = np.sum(consumed_and_discounted_energy)

        return consumed_and_discounted_energy


if __name__ == "__main__":
    import json

    config = json.load(open("../configs/config_fdr.json"))
    config["flexible_demand_response"]["planning_horizon"] = 4

    fdr = FlexibleDemandResponse(**config["flexible_demand_response"])

    print(fdr.patience_weighting)
    exit()
    fdr.episode.values[0:100] = np.zeros_like(fdr.episode.values[0:100])
    fdr.episode.values[0:3] = 1
    fdr.update_state()
    actions = [-1] * 20
    for action in actions:
        print(f"Action: {action}, Previous State: {fdr.state}")
        fdr.step(action)
        print(f"State: {fdr.state}, Reward Cache: {fdr.reward_cache}")
