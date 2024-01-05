import numpy as np
from src.environment.components.base.component import Component


class ThermostaticallyControlledLoad(Component):
    """
        Represents a thermostatically controlled load.

        This class inherits from the Component class.
        """

    def __init__(self,
                 initial_indoor_temperature: float,
                 initial_building_mass_temperature: float,
                 thermal_mass_air: float,
                 thermal_mass_building: float,
                 unintentional_heat_gain: float,
                 degree_generated_by_kw: float,
                 nominal_power: float,
                 minimal_temperature: float,
                 maximal_temperature: float,
                 penalty_factor: float,
                 ):
        """
        Initializes the ThermostaticallyControlledLoad.

        Args:
            initial_indoor_temperature (float): The initial indoor temperature in degree Celsius.
            initial_building_mass_temperature (float): The initial building mass temperature in degree Celsius.
            thermal_mass_air (float): The thermal mass of the air.
            thermal_mass_building (float): The thermal mass of the building.
            unintentional_heat_gain (float): The unintentional heat gain in degree Celsius/min.
            degree_generated_by_kw (float): The degree generated by kW*resolution.
            nominal_power (float): The nominal power in kW.
            minimal_temperature (float): The minimal temperature.
            maximal_temperature (float): The maximal temperature.
            penalty_factor (float): The penalty factor for the reward.
        """
        super().__init__()
        self.initial_indoor_temperature = initial_indoor_temperature

        self.indoor_temperature = initial_indoor_temperature
        self.building_mass_temperature = initial_building_mass_temperature
        self.minimal_temperature = minimal_temperature
        self.maximal_temperature = maximal_temperature
        self.desired_temperature = (self.minimal_temperature + self.maximal_temperature) / 2

        self.thermal_mass_air = thermal_mass_air
        self.thermal_mass_building = thermal_mass_building
        self.unintentional_heat_gain = unintentional_heat_gain
        self.degree_generated_by_kw = degree_generated_by_kw
        self.nominal_power = nominal_power
        self.penalty_factor = penalty_factor

    def reset(self, initial_indoor_temperature:float = None):
        """
        Reset the thermostatically controlled load.

        Args:
            initial_indoor_temperature (float): The initial indoor temperature in degree Celsius.
        """
        self.indoor_temperature = self.initial_indoor_temperature if initial_indoor_temperature is None else initial_indoor_temperature
        self.building_mass_temperature = self.indoor_temperature
        self.update_state()

    def step(self, action: float, outdoor_temperature: float):
        """
        Perform a step in the environment.

        Args:
            action (float): The action to be taken [0, 1]. -1 is full cooling, 1 is full heating.
            outdoor_temperature (float): The outdoor temperature in degree Celsius.
        """
        if self.indoor_temperature > self.maximal_temperature:
            action = -1
        elif self.indoor_temperature < self.minimal_temperature:
            action = 1

        self.building_mass_temperature += 1 / self.thermal_mass_building * (
                self.indoor_temperature - self.building_mass_temperature)

        self.indoor_temperature += 1 / self.thermal_mass_air * (outdoor_temperature - self.indoor_temperature)
        self.indoor_temperature += 1 / self.thermal_mass_building * (
                self.building_mass_temperature - self.indoor_temperature)
        self.indoor_temperature += action * self.nominal_power * self.degree_generated_by_kw
        self.indoor_temperature += self.unintentional_heat_gain

        self.update_state()
        self.update_reward_cache(action)

    def update_state(self):
        """
        Update the state.
        """
        self.state = self.indoor_temperature

    def update_reward_cache(self, action):
        """
        Update the reward cache with the action.

        Args:
            action (float): The action performed.
        """
        self.reward_cache["consumed_energy"] = np.abs(action) * self.nominal_power
        self.reward_cache["indoor_temperature"] = self.indoor_temperature


if __name__ == "__main__":
    import json

    config = json.load(open("../configs/config_tcl.json"))
    thermostatically_controlled_load = ThermostaticallyControlledLoad(**config["thermostatically_controlled_load"])

    outdoor_temperatures = [20, -1, 25, 20, 20, 20]
    actions = [0, 0, 0, 1, 1, 0]
    for action, outdoor_temperature in zip(actions, outdoor_temperatures):
        thermostatically_controlled_load.step(action, outdoor_temperature)
        print(f"Action: {action}, Outdoor Temperature: {outdoor_temperature}")
        print(
            f"State: {thermostatically_controlled_load.state}, Reward Cache: {thermostatically_controlled_load.reward_cache}")
