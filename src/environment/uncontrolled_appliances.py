from datetime import datetime

import pandas as pd


class UncontrolledAppliances:
    def __init__(self):
        self.file = '../../data/minutely/uncontrolled_appliances.h5'
        self.episode = None

        self.reward_cache = {}

    def set_episode(self, episode: int):
        self.episode = pd.read_hdf(self.file, key=f'eps_{episode}')

    def get_values(self, start: datetime, end: datetime = None):
        if end is None:
            end = start
        return self.episode.loc[start:end]

    def step(self, time: datetime):
        """
        Returns the power generated at a given time in Watts
        :param time: datetime

        :return: float
        """
        consumed_power = self.get_values(time)["power"].values
        self.reward_cache["L_t"] = consumed_power


if __name__ == "__main__":
    gen = UncontrolledAppliances()
    gen.set_episode(0)

    gen.step(datetime(2007, 1, 1, 18, 0, 0))
    print(gen.reward_cache)
