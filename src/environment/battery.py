class Battery:
    def __init__(self,
                 capacity,
                 charge_coefficient,
                 discharge_coefficient,
                 dissipation,
                 lossC,
                 rateC,
                 maxDD,
                 chargeE,
                 tmax
                 ):
        self.capacity = capacity
        self.charge_coefficient = charge_coefficient
        self.discharge_coefficient = discharge_coefficient

        self.dissipation = dissipation  # dissipation coefficient of the battery
        self.lossC = lossC  # charge loss
        self.rateC = rateC  # charging rate
        self.maxDD = maxDD  # maximum power that the battery can deliver per timestep
        self.tmax = tmax  # maxmum charging time
        self.chargeE = chargeE  # Energy given to the battery to charge
        self.RC = 0  # remaining capacity
        self.ct = 0  # Charging step

    def charge(self, E):
        empty = self.capacity - self.RC
        if empty <= 0:
            return E
        else:
            self.RC += self.rateC * E
            leftover = self.RC - self.capacity
            self.RC = min(self.capacity, self.RC)
            return max(leftover, 0)

    def supply(self, E):
        remaining = self.RC
        self.RC -= E * self.useD
        self.RC = max(self.RC, 0)
        return min(E, remaining)

    def dissipate(self):
        self.RC = self.RC * math.exp(- self.dissipation)

    @property
    def SoC(self):
        return self.RC / self.capacity
