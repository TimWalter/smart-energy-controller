import numpy as np


class Battery:
    def __init__(self,
                 capacitiy,
                 charge_rate_max,
                 discharge_rate_max,
                 round_trip_efficiency,
                 self_discharge_rate,
                 initial_charge
                 ):
        self.capacity = capacitiy
        self.charge_rate_max = charge_rate_max
        self.discharge_rate_max = discharge_rate_max
        self.round_trip_efficiency_sqrt = np.sqrt(round_trip_efficiency)
        self.self_discharge_rate = self_discharge_rate

        self.charge = initial_charge

        self.reward_cache = {}

    def step(self, action):
        lower_bound = self.charge * self.self_discharge_rate * self.round_trip_efficiency_sqrt
        upper_bound = (self.capacity - self.charge * self.self_discharge_rate) / self.round_trip_efficiency_sqrt
        actual_action = np.clip(action, lower_bound, upper_bound)

        charging_rate = min(max(actual_action, 0), self.charge_rate_max)
        discharge_rate = min(max(-actual_action, 0), self.discharge_rate_max)

        self.charge = (self.charge * self.self_discharge_rate
                       + charging_rate * self.round_trip_efficiency_sqrt
                       - discharge_rate / self.round_trip_efficiency_sqrt)
        
        self.reward_cache = {
            "C_t": charging_rate,
            "D_t": discharge_rate,
        }

    @property
    def state_of_charge(self):
        return self.charge / self.capacity
