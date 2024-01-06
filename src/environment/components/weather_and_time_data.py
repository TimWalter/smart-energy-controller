import numpy as np

from src.environment.components.base.component import Component
from src.environment.components.base.data_loader import DataLoader


class WeatherAndTimeData(Component, DataLoader):
    """
        Class to provide weather and time data.

        This class inherits from the Component and DataLoader classes.
    """

    def __init__(self, resolution: str):
        """
        Initializes the WeatherAndTimeData.

        Args:
            episode (int): The current episode. Defaults to 0.
        """
        Component.__init__(self)
        DataLoader.__init__(self, file=f'../data/{resolution}/weather_and_time.h5', resolution=resolution)

    def reset(self, episode: int):
        """
        Resets the external electricity supply.
        """
        self.set_episode(episode)
        self.update_state()

    def step(self):
        """
        Performs a step in the external electricity supply.
        """
        self.step_time()
        self.update_state()
        self.update_reward_cache()

    def update_state(self):
        """
        Updates day of year, hour of day, and weather data.
        """
        self.state = np.array([self.time.month])
        self.state = np.concatenate((self.state, self.get_values(self.time).values[0]))

    def update_reward_cache(self):
        pass


if __name__ == "__main__":
    weather_and_time_data = WeatherAndTimeData(0)

    print(f"State: {weather_and_time_data.state}, Reward Cache: {weather_and_time_data.reward_cache}")
    weather_and_time_data.step()
    print(f"State: {weather_and_time_data.state}, Reward Cache: {weather_and_time_data.reward_cache}")
