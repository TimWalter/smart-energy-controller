from src.environment.components.base.component import Component
from src.environment.components.base.data_loader import DataLoader


class HouseholdEnergyDemand(Component, DataLoader):
    """
    Represents a household energy demand.

    This class inherits from the Component and DataLoader classes.
    """

    def __init__(self, episode: int = 0):
        """
        Initializes the HouseholdEnergyDemand.

        Args:
            episode (int): The current episode. Defaults to 0.
        """
        Component.__init__(self)
        DataLoader.__init__(self, file='../data/minutely/household_energy_demand.h5')
        self.set_episode(episode)

        self.update_state()

    def step(self):
        """
        Performs a step in the household energy demand.
        """
        self.step_time()
        self.update_state()
        self.update_reward_cache()

    def update_state(self):
        """
        Updates the household energy demand.
        """
        self.state = self.get_values(self.time)["energy"].values[0]  # in kW

    def update_reward_cache(self):
        """
        Updates the reward cache of the household energy demand.
        """
        self.reward_cache["household_energy_demand"] = self.state


if __name__ == "__main__":
    household_energy_demand = HouseholdEnergyDemand(0)

    print(f"State: {household_energy_demand.state}, Reward Cache: {household_energy_demand.reward_cache}")
    household_energy_demand.step()
    print(f"State: {household_energy_demand.state}, Reward Cache: {household_energy_demand.reward_cache}")
