class TCL:
    def __init__(self,
                 T_initial_indoor,
                 T_initial_building_mass,
                 thermal_mass_air,
                 thermal_mass_building,
                 unintentional_heat_gain,
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
        self.nominal_power = nominal_power

    def step(self, action, T_outdoor):
        if self.T_indoor > self.T_max:
            action = 0
        elif self.T_indoor < self.T_min:
            action = 1

        self.T_building_mass += 1 / self.thermal_mass_building * (
                self.T_indoor - self.T_building_mass)

        self.T_indoor += 1 / self.thermal_mass_air * (T_outdoor - self.T_indoor)
        self.T_indoor += 1 / self.thermal_mass_building * (self.T_building_mass - self.T_indoor)
        self.T_indoor += action * self.nominal_power
        self.T_indoor += self.unintentional_heat_gain

    @property
    def state_of_charge(self):
        return (self.T_indoor - self.T_min) / (self.T_max - self.T_min)
