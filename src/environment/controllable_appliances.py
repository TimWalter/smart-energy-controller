from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class ControllableAppliances:
    def __init__(self, planning_horizon, patience):
        self.file = '../../data/minutely/controllable_appliances.h5'
        self.episode = None

        self.planning_horizon = planning_horizon
        self.half_horizon = timedelta(minutes=planning_horizon // 2)
        self.patience = patience

        self.reward_cache = {}

    def set_episode(self, episode: int):
        self.episode = pd.read_hdf(self.file, key=f'eps_{episode}')

    def get_values(self, start: datetime, end: datetime = None):
        if end is None:
            end = start
        return self.episode.loc[start:end]

    def set_values(self, values, start: datetime, end: datetime = None):
        if end is None:
            end = start
        self.episode.loc[start:end] = values

    def step(self, time: datetime, action: np.ndarray):
        current_window = self.get_values(time - self.half_horizon, time + self.half_horizon)

        coins = np.random.uniform(0, 1, action.shape)
        probas = np.exp(-1 / self.patience * np.abs((time - current_window.index).total_seconds() // 60))
        executed_actions = coins < probas

        power = np.sum(current_window["power"].values[executed_actions])

        current_window.loc[:, "power"] = current_window["power"].values * (executed_actions == 0)
        self.set_values(current_window, time - self.half_horizon, time + self.half_horizon)

        # Delay beyond time_horizon results in deterministic execution
        power += current_window["power"].values[-1]

        self.reward_cache["s_{a,t}"] = power


if __name__ == "__main__":
    gen = ControllableAppliances(4, 10)

    gen.set_episode(0)

    gen.step(datetime(2007, 1, 1, 0, 20, 0), np.array([1, 1, 1, 1, 1]))
    print(gen.get_values(datetime(2007, 1, 1, 0, 19, 0), datetime(2007, 1, 1, 0, 23, 0)))
