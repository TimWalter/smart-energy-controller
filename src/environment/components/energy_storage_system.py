import numpy as np

from src.environment.components.base.component import Component


class EnergyStorageSystem(Component):
    def __init__(self,
                 capacity: float,
                 maximum_charge_rate_in_kw: float,
                 maximum_discharge_rate_in_kw: float,
                 round_trip_efficiency: float,
                 self_discharge_rate: float,
                 initial_charge: float,
                 ):
        """
        From now on units are dropped and expected to be in kW*resolution for energy and kW for power.

        Args:
            capacity (float): The capacity of the energy storage system in kW*resolution.
            maximum_charge_rate_in_kw (float): The maximum charge rate of the energy storage system in kW.
            maximum_discharge_rate_in_kw (float): The maximum discharge rate of the energy storage system in kW.
            round_trip_efficiency (float): The round trip efficiency of the energy storage system.
            self_discharge_rate (float): The self discharge rate of the energy storage system.
            initial_charge (float): The initial charge of the energy storage system in kW*resolution.
        """

        super().__init__()
        self.initial_charge = initial_charge

        self.capacity = capacity
        self.maximum_charge_rate = maximum_charge_rate_in_kw
        self.maximum_discharge_rate = maximum_discharge_rate_in_kw
        self.round_trip_efficiency_sqrt = np.sqrt(round_trip_efficiency)
        self.self_discharge_rate = self_discharge_rate
        self.charge = initial_charge

    def reset(self, initial_charge:float = None):
        """
        Reset the energy storage system.
        """
        self.charge = self.initial_charge if initial_charge is None else initial_charge
        self.update_state()

    def step(self, action: float):
        """
        Perform a step in the environment.

        Args:
            action (np.ndarray): The action to be taken [-1, 1]. -1 is full discharge, 1 is full charge.
            The action is clipped to the possible charge and discharge rates.
        """

        possible_discharge = np.max([self.charge - self.self_discharge_rate, 0]) * self.round_trip_efficiency_sqrt
        discharge_rate = np.clip(-action * self.maximum_discharge_rate, 0, possible_discharge)

        possible_charge = (self.capacity - self.charge + self.self_discharge_rate) / self.round_trip_efficiency_sqrt
        charge_rate = np.clip(action * self.maximum_charge_rate, 0, possible_charge)

        self.charge = np.max([self.charge - self.self_discharge_rate
                              + charge_rate * self.round_trip_efficiency_sqrt
                              - discharge_rate / self.round_trip_efficiency_sqrt, 0])

        self.update_state()
        self.update_reward_cache(charge_rate, discharge_rate)

    def update_state(self):
        """
        Update the state.
        """
        self.state = self.charge

    def update_reward_cache(self, charge_rate: float, discharge_rate: float):
        """
        Update the reward cache with the charge and discharge rate.

        Args:
            charge_rate (float): The charge rate in kW.
            discharge_rate (float): The discharge rate in kW.
        """
        self.reward_cache["consumed_energy"] = charge_rate - discharge_rate


if __name__ == "__main__":
    import json

    config = json.load(open("../configs/config_ess.json"))
    energy_storage_system = EnergyStorageSystem(**config["energy_storage_system"])
    charges = [0, 0, 0, 10, 1, 808]
    actions = [0, -1, 1, 1, -1, 1]
    for action, charge in zip(actions, charges):
        energy_storage_system.charge = charge
        energy_storage_system.step(action)
        print(f"Action: {action}, Charge: {charge}")
        print(f"State: {energy_storage_system.state}, Reward Cache: {energy_storage_system.reward_cache}")
