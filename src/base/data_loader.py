from abc import ABC
from datetime import datetime

import numpy as np
import pandas as pd


class BaseDataLoader(ABC):
    def __init__(self, file: str):
        self.file = file
        self.episode = None
        self.time = None

    def set_episode(self, episode: int):
        self.episode = pd.DataFrame(pd.read_hdf(self.file, key=f'eps_{episode}'))
        self.time = self.episode.index[0]

    def get_values(self, start: datetime, end: datetime = None):
        if end is None:
            end = start
        return self.episode.loc[start:end]

    def set_values(self, values, start: datetime, end: datetime = None, column: str = None):
        if end is None:
            end = start

        start_index = int(np.max([0, (self.episode.index[0] - start).total_seconds()])) // 60
        end_index = int((self.episode.index[-1] - end).total_seconds() // 60)
        if end_index >= 0:
            end_index = None

        try:
            values = values[start_index:end_index]
        except TypeError:
            print(f"Start Index: {start_index} {type(start_index)}")
            print(f"End Index: {end_index} {type(end_index)}")
            raise TypeError
        try:
            if column is None:
                self.episode.loc[start:end] = values
            else:
                self.episode.loc[start:end, column] = values
        except ValueError:
            print(f"Values: {values}")
            print(f"Start: {start}")
            print(f"End: {end}")
            print(f"Actual Start: {self.episode.index[0]}")
            print(f"Actual End: {self.episode.index[-1]}")
            print(f"Start Index: {start_index}")
            print(f"End Index: {end_index}")
            raise ValueError

    def step_time(self):
        self.time += pd.Timedelta(minutes=1)
