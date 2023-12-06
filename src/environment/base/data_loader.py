from abc import ABC, abstractmethod
from datetime import datetime

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

    def set_values(self, values, start: datetime, end: datetime = None):
        if end is None:
            end = start
        self.episode.loc[start:end] = values

    def step_time(self):
        self.time += pd.Timedelta(minutes=1)
