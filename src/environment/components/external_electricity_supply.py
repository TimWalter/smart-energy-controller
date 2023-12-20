from src.environment.components.base.component import Component
from src.environment.components.base.data_loader import DataLoader


class ExternalElectricitySupply(Component, DataLoader):
    """
    Represents an external electricity supply.

    This class inherits from the Component and DataLoader classes.
    """

    def __init__(self, episode: int = 0):
        """
        Initializes the ExternalElectricitySupply.

        Args:
            episode (int): The current episode. Defaults to 0.
        """
        Component.__init__(self)
        DataLoader.__init__(self, file='../data/minutely/carbon_intensity.h5')
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
        Updates carbon intensity.
        """
        self.state = self.get_values(self.time)["Carbon Intensity gCO₂eq/kWmin (direct)"].values[0]

    def update_reward_cache(self):
        """
        Updates the reward cache of the external electricity supply.
        """
        self.reward_cache["carbon_intensity"] = self.state


if __name__ == "__main__":
    external_electricity_supply = ExternalElectricitySupply(0)

    print(f"State: {external_electricity_supply.state}, Reward Cache: {external_electricity_supply.reward_cache}")
    external_electricity_supply.step()
    print(f"State: {external_electricity_supply.state}, Reward Cache: {external_electricity_supply.reward_cache}")
