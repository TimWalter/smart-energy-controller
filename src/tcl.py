from dataclasses import dataclass

import numpy as np

from src.base.base_component import BaseComponent


@dataclass
class TCLParameters:
    T_initial_indoor: float = 20
    T_initial_building_mass: float = 20
    thermal_mass_air: float = 1500
    thermal_mass_building: float = 600
    unintentional_heat_gain: float = 0.1 / 60
    coefficient_T_by_P: float = 0.5 / 60  # defines C°/kWmin
    nominal_power: float = 30  # how much kW the heater has
    T_min: float = 18
    T_max: float = 23


class TCL(BaseComponent):
    def __init__(self,
                 T_initial_indoor,
                 T_initial_building_mass,
                 thermal_mass_air,
                 thermal_mass_building,
                 unintentional_heat_gain,
                 coefficient_T_by_P,
                 nominal_power,
                 T_min,
                 T_max
                 ):
        self.T_indoor = T_initial_indoor
        self.T_building_mass = T_initial_building_mass

        self.T_min = T_min
        self.T_max = T_max

        self.thermal_mass_air = thermal_mass_air
        self.thermal_mass_building = thermal_mass_building
        self.unintentional_heat_gain = unintentional_heat_gain
        self.coefficient_T_by_P = coefficient_T_by_P
        self.nominal_power = nominal_power

        self.update_state()
        super().__init__(initial_state=self.state)
        self.reward_cache = {"L_{TCL}": nominal_power}

    def step(self, action, T_outdoor):
        # rescale action from [-1,1] to [0,1]
        action = (action[0] + 1) / 2
        if self.T_indoor > self.T_max:
            action = 0
        elif self.T_indoor < self.T_min:
            action = 1

        self.T_building_mass += 1 / self.thermal_mass_building * (
                self.T_indoor - self.T_building_mass)

        self.T_indoor += 1 / self.thermal_mass_air * (T_outdoor - self.T_indoor)
        self.T_indoor += 1 / self.thermal_mass_building * (self.T_building_mass - self.T_indoor)
        self.T_indoor += action * self.nominal_power * self.coefficient_T_by_P
        self.T_indoor += self.unintentional_heat_gain

        self.update_reward_cache(action)
        self.update_state()

    def update_state(self):
        self.state = np.array([(self.T_indoor - self.T_min) / (self.T_max - self.T_min)], dtype=np.float32)

    def update_reward_cache(self, action):
        self.reward_cache["a_{tcl,t}"] = action


if __name__ == "__main__":
    tcl = TCL(**TCLParameters().__dict__)
    print(tcl.state)
    tcl.step(1, -1)
    print(tcl.state)
    print(tcl.reward_cache)
