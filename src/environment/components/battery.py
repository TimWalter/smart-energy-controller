from dataclasses import dataclass

import numpy as np

from src.environment.components.base.base_component import BaseComponent


@dataclass
class BatteryParameters:
    capacity: float = 3000  # in kWmin
    charge_rate_max: float = 60  # in kW
    discharge_rate_max: float = 60
    round_trip_efficiency: float = 0.95
    self_discharge_rate: float = 0.9999
    initial_charge: float = 60


class Battery(BaseComponent):
    def __init__(self,
                 capacity: float,
                 charge_rate_max: float,
                 discharge_rate_max: float,
                 round_trip_efficiency: float,
                 self_discharge_rate: float,
                 initial_charge: float
                 ):
        super().__init__(normalise=False,
                         max_state=1,
                         min_state=0,
                         )
        self.capacity = capacity
        self.charge_rate_max = charge_rate_max
        self.discharge_rate_max = discharge_rate_max
        self.round_trip_efficiency_sqrt = np.sqrt(round_trip_efficiency)
        self.self_discharge_rate = self_discharge_rate

        self.charge = initial_charge

        self.update_state()

    def step(self, action):
        possible_discharge = self.charge * self.self_discharge_rate * self.round_trip_efficiency_sqrt
        possible_charge = (self.capacity - self.charge * self.self_discharge_rate) / self.round_trip_efficiency_sqrt

        bound_action = np.clip(action[0],
                               -possible_discharge / self.discharge_rate_max,
                               possible_charge / self.charge_rate_max
                               )

        charging_rate = max(bound_action, 0) * self.charge_rate_max
        discharge_rate = max(-bound_action, 0) * self.discharge_rate_max

        self.charge = (self.charge * self.self_discharge_rate
                       + charging_rate * self.round_trip_efficiency_sqrt
                       - discharge_rate / self.round_trip_efficiency_sqrt)

        self.update_reward_cache(charging_rate, discharge_rate)
        self.update_state()

    def update_state(self):
        self.state = self.charge / self.capacity

    def update_reward_cache(self, charging_rate, discharge_rate):
        self.reward_cache["C_t"] = charging_rate
        self.reward_cache["D_t"] = discharge_rate


if __name__ == "__main__":
    battery = Battery(**BatteryParameters().__dict__)
    battery.charge = 2950
    print(battery.state)
    battery.step([1])
    print(battery.state)
    print(battery.reward_cache)
