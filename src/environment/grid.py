class Grid:
    def __init__(self):
        down_reg_df = pd.read_csv("../../data/down_regulation.csv")
        up_reg_df = pd.read_csv("../../data/up_regulation.csv")
        down_reg = np.array(down_reg_df.iloc[:, -1]) / 10
        up_reg = np.array(up_reg_df.iloc[:, -1]) / 10
        self.buy_prices = down_reg
        self.sell_prices = up_reg
        self.time = 0

    def sell(self, E):
        return self.sell_prices[self.time] * E

    def buy(self, E):
        return -self.buy_prices[self.time] * E - QUADRATIC_PRICE * E ** 2 - FIXED_COST

    #
    # def get_price(self,time):
    #     return self.prices[time]

    def set_time(self, time):
        self.time = time