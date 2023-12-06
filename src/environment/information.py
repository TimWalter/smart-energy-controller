from datetime import datetime

from src.environment.base.base_component import BaseComponent
from src.environment.base.data_loader import BaseDataLoader


class Information(BaseComponent, BaseDataLoader):

    def __init__(self, episode: int = 0):
        BaseDataLoader.__init__(self, file='../../data/minutely/information.h5')
        self.set_episode(episode)

        self.update_state()
        BaseComponent.__init__(self, initial_state=self.state)

    def step(self):
        self.update_reward_cache()
        self.step_time()
        self.update_state()

    def update_state(self):
        self.state = self.get_values(self.time)

    def update_reward_cache(self):
        pass


if __name__ == "__main__":
    info = Information(0)

    info.step()
    print(info.reward_cache)
    print(info.state)
