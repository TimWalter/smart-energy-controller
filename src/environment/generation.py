from datetime import datetime

import pandas as pd


class Generation:
    def __init__(self):
        self.file = '../../data/minutely/pv.h5'
        self.episode = None

    def set_episode(self, episode: int):
        self.episode = pd.read_hdf(self.file, key=f'eps_{episode}')

    def power(self, time: datetime):
        """
        Returns the power generated at a given time in Watts
        :param time: datetime

        :return: float
        """
        return self.episode.iloc[self.episode.index.get_indexer([pd.to_datetime(time)], method='nearest')]
