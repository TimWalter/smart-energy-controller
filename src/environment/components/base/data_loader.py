from abc import ABC
from datetime import datetime

import pandas as pd


class DataLoader(ABC):
    def __init__(self, file: str, resolution: str):
        self.file = file
        self.episode = None
        self.time = None
        self.resolution = resolution
        self.second_to_resolution = 60 if resolution == "minutely" else 3600
        self.one_timestep_delta = pd.Timedelta(seconds=self.second_to_resolution)

    def set_episode(self, episode: int):
        try:
            self.episode = pd.DataFrame(pd.read_hdf(self.file, key=f'eps_{episode}'))
        except FileNotFoundError:
            self.file = "../" + self.file
            self.set_episode(episode)
        self.time = self.episode.index[0]

    def get_values(self, start: datetime, end: datetime = None):
        if end is None:
            end = start
        return self.episode.loc[start:end]

    def set_values(self, values, start: datetime, end: datetime = None, column: str = None):
        if end is None:
            end = start

        start_index = 0
        end_index = None
        if start < self.episode.index[0]:
            start_index = int((self.episode.index[0] - start).total_seconds() // self.second_to_resolution)
        if end > self.episode.index[-1]:
            end_index = int((self.episode.index[-1] - end).total_seconds() // self.second_to_resolution)

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
        self.time += self.one_timestep_delta
