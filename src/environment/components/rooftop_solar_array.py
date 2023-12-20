from src.environment.components.base.component import Component
from src.environment.components.base.data_loader import DataLoader


class RooftopSolarArray(Component, DataLoader):
    """
    Represents a rooftop solar array.

    This class inherits from the Component and DataLoader classes.
    """

    def __init__(self, episode: int = 0):
        """
        Initializes the RooftopSolarArray.

        Args:
            episode (int): The current episode. Defaults to 0.
        """
        Component.__init__(self)
        DataLoader.__init__(self, file='../data/minutely/rooftop_solar_array.h5')
        self.set_episode(episode)

        self.update_state()

    def step(self):
        """
        Performs a step in the rooftop solar array.
        """
        self.step_time()
        self.update_state()
        self.update_reward_cache()

    def update_state(self):
        """
        Updates the energy generation of the rooftop solar array.
        """
        self.state = self.get_values(self.time)["energy"].values[0]  # in kW

    def update_reward_cache(self):
        """
        Updates the reward cache of the rooftop solar array.
        """
        self.reward_cache["rooftop_solar_generation"] = self.state


if __name__ == "__main__":
    rooftop_solar_array = RooftopSolarArray(0)

    print(f"State: {rooftop_solar_array.state}, Reward Cache: {rooftop_solar_array.reward_cache}")
    rooftop_solar_array.step()
    print(f"State: {rooftop_solar_array.state}, Reward Cache: {rooftop_solar_array.reward_cache}")
