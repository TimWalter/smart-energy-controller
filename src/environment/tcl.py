class TCL:
    """
    Simulates an invidual TCL
    """

    def __init__(self, ca, cm, q, P, Tmin=TCL_TMIN, Tmax=TCL_TMAX):
        self.ca = ca
        self.cm = cm
        self.q = q
        self.P = P
        self.Tmin = Tmin
        self.Tmax = Tmax

        # Added for clarity
        self.u = 0

    def set_T(self, T, Tm):
        self.T = T
        self.Tm = Tm

    def control(self, ui=0):
        # control TCL using u with respect to the backup controller
        if self.T < self.Tmin:
            self.u = 1
        elif self.Tmin < self.T < self.Tmax:
            self.u = ui
        else:
            self.u = 0

    def update_state(self, T0):
        # update the indoor and mass temperatures according to (22)
        for _ in range(5):
            self.T += self.ca * (T0 - self.T) + self.cm * (self.Tm - self.T) + self.P * self.u + self.q
            self.Tm += self.cm * (self.T - self.Tm)
            if self.T >= self.Tmax:
                break

    """ 
    @property allows us to write "tcl.SoC", and it will
    run this function to get the value
    """

    @property
    def SoC(self):
        return (self.T - self.Tmin) / (self.Tmax - self.Tmin)
