from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from src.base.base_component import BaseComponent
from src.base.data_loader import BaseDataLoader


@dataclass
class AdaptiveConsumptionParameters:
    planning_horizon: int = 60 * 12
    patience: float = 100


class AdaptiveConsumption(BaseComponent, BaseDataLoader):
    def __init__(self, planning_horizon: int, patience: float, episode: int = 0):
        self.planning_horizon = planning_horizon
        self.patience = patience
        self.timedelta = timedelta(minutes=planning_horizon)

        BaseDataLoader.__init__(self, file='../data/minutely/adaptive_consumption.h5')
        self.set_episode(episode)

        self.update_state()
        BaseComponent.__init__(self, initial_state=self.state)

        self.desired_behaviour = np.concatenate([np.ones(self.planning_horizon + 1), np.zeros(self.planning_horizon)])
        self.weighting = np.exp(
            -1 / self.patience * np.abs((np.arange(-self.planning_horizon, self.planning_horizon + 1))))

    def step(self, action: np.ndarray):
        weighted_action = (2 * action - 1) * self.weighting

        probas = np.clip(self.desired_behaviour + weighted_action, 0, 1)
        coins = np.random.uniform(0, 1, action.shape)

        executed_actions = coins < probas

        power = np.sum(self.state[executed_actions])

        # Set power to 0 for executed actions
        self.state = self.state * (executed_actions == 0)
        self.set_values(self.state, self.time - self.timedelta, self.time + self.timedelta, "power")

        # Delay beyond time_horizon results in deterministic execution
        power += self.state[0]

        self.update_reward_cache(power)
        self.step_time()
        self.update_state()

    def update_state(self):
        self.state = self.get_values(self.time - self.timedelta, self.time + self.timedelta)  # in kW
        rows_to_add = 2 * self.planning_horizon + 1 - self.state.shape[0]

        if rows_to_add > 0:
            # Create a new DataFrame with zeros and the correct index
            if self.time - self.timedelta < self.state.index[0]:
                # If time is before the start of the state DataFrame, prepend the rows
                zero_data = pd.DataFrame({"power": [0] * rows_to_add},
                                         index=pd.date_range(end=self.state.index[0] - pd.Timedelta(minutes=1),
                                                             periods=rows_to_add, freq='min'))
                self.state = pd.concat([zero_data, self.state])
            else:
                # If time is after the end of the state DataFrame, append the rows
                zero_data = pd.DataFrame({"power": [0] * rows_to_add},
                                         index=pd.date_range(start=self.state.index[-1] + pd.Timedelta(minutes=1),
                                                             periods=rows_to_add, freq='min'))
                self.state = pd.concat([self.state, zero_data])

        # Need np.array so only values remain
        self.state = np.array(self.state["power"], dtype=np.float32)

    def update_reward_cache(self, power):
        self.reward_cache["s_{a,t}"] = power


if __name__ == "__main__":
    gen = AdaptiveConsumption(**AdaptiveConsumptionParameters().__dict__)
    print(gen.state)
    for i in range(21):
        if i == 20:
            print(gen.state)
        gen.step(np.array([0, 0, 0, 1, 1]))
    print(gen.state)
    print(gen.reward_cache)
