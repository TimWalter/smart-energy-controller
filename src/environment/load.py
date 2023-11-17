class Load:
    def __init__(self, price_sens, base_load, max_v_load):
        self.price_sens = price_sens
        self.base_load = base_load
        self.max_v_load = max_v_load
        self.response = 0

    def react(self, price_tier):
        self.response = self.price_sens * (price_tier - 2)
        if self.response > 0 and self.price_sens > 0.1:
            self.price_sens -= 0.1

    def load(self, time_day):
        # print(self.response)
        return max(self.base_load[time_day] - self.max_v_load * self.response, 0)