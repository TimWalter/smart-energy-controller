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

    def __init__(self, planning_horizon: int, deterministic: bool, patience: float, episode: int = 0):
        """
        Initializes the FlexibleDemandResponse. All energy expenditures are in kWmin.

        Args:
            planning_horizon (int): The planning horizon.
            deterministic (bool): If the response is deterministic.
            patience (float): The patience factor.
            episode (int): The current episode. Defaults to 0.
        """
        Component.__init__(self)
        DataLoader.__init__(self, file='../data/minutely/flexible_demand_response.h5')
        self.set_episode(episode)

        self.planning_horizon = planning_horizon
        self.deterministic = deterministic
        self.patience = patience

        self.timedelta = timedelta(minutes=planning_horizon)
        self.schedule = np.concatenate([np.ones(self.planning_horizon + 1), np.zeros(self.planning_horizon)])
        if self.deterministic:
            self.weighting = np.ones_like(self.schedule)
        else:
            self.weighting = np.exp(-1 / self.patience * np.abs(np.arange(len(self.schedule)) - self.planning_horizon))
            self.weighting *= np.concatenate([-1*np.ones(self.planning_horizon + 1), np.ones(self.planning_horizon)])

        self.update_state()

    def step(self, actions: np.ndarray):
        """
        Perform a step in the environment.

        Args:
            actions (np.ndarray): The actions are in [0, 1].
            if usage was scheduled in the past 1 corresponds to maximum delay,
            if usage is scheduled in the future 1 corresponds to maximum expedite.
        """
        execution_probability = np.clip(self.schedule + actions * self.weighting, 0, 1)
        coins = np.random.uniform(0, 1, actions.shape)

        executed_actions = coins < execution_probability

        power = np.sum(self.state[executed_actions])

        # Set power to 0 for executed actions
        self.state[executed_actions] = 0
        self.set_values(self.state, self.time - self.timedelta, self.time + self.timedelta, "energy")

        # Delay beyond time_horizon results in deterministic execution
        power += self.state[0]

        self.update_reward_cache(power)
        self.step_time()
        self.update_state()

    def update_state(self):
        """
        Update the the flexible demand response window.
        """
        self.state = self.get_values(self.time - self.timedelta, self.time + self.timedelta)  # in kW

        rows_to_add = len(self.schedule) - self.state.shape[0]

        # Ensure zero padding
        if rows_to_add > 0:
            if self.time - self.timedelta < self.state.index[0]:
                # If time is before the start of the state DataFrame, prepend the rows
                zero_data = pd.DataFrame({"energy": [0] * rows_to_add},
                                         index=pd.date_range(end=self.state.index[0] - pd.Timedelta(minutes=1),
                                                             periods=rows_to_add, freq='min'))
                self.state = pd.concat([zero_data, self.state])
            else:
                # If time is after the end of the state DataFrame, append the rows
                zero_data = pd.DataFrame({"energy": [0] * rows_to_add},
                                         index=pd.date_range(start=self.state.index[-1] + pd.Timedelta(minutes=1),
                                                             periods=rows_to_add, freq='min'))
                self.state = pd.concat([self.state, zero_data])

        self.state = self.state["energy"].values

    def update_reward_cache(self, power):
        """
        Update the reward cache with the power.

        Args:
            power (float): The power in kW.
        """
        self.reward_cache["flexible_demand_response"] = power


if __name__ == "__main__":
    import json

    config = json.load(open("../config.json"))
    config["flexible_demand_response"]["planning_horizon"] = 3

    fdr = FlexibleDemandResponse(**config["flexible_demand_response"])
    for i in range(20):
        fdr.step(np.zeros(7))
    actions = [
        np.array([0, 0, 0, 0, 0, 0, 0]),
        np.array([0, 0, 0, 1, 1, 0, 0]),
        np.array([0, 0, 1, 0, 1, 1, 1]),
        np.array([1, 0, 0, 0, 0, 0, 0]),
        np.array([0, 0, 0, 0, 0, 0, 1]),
    ]
    for action in actions:
        print(f"Action: {action}, Previous State: {fdr.state}")
        fdr.step(action)
        print(f"State: {fdr.state}, Reward Cache: {fdr.reward_cache}")
