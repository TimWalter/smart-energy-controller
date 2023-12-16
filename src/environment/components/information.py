from src.environment.components.base.base_component import BaseComponent
from src.environment.components.base.data_loader import BaseDataLoader


class Information(BaseComponent, BaseDataLoader):

    def __init__(self, episode: int = 0, episode_length: int = None, normalise: bool = False):
        BaseComponent.__init__(self,
                               normalise=normalise,
                               max_state=np.array({
                                                      "G(i)": 1082.1,
                                                      "H_sun": 64.41,
                                                      "T2m": 35.13,
                                                      "WS10m": 12.76
                                                  }.values()),
                               min_state=np.array({
                                                      "G(i)": 0.0,
                                                      "H_sun": 0.0,
                                                      "T2m": -10.43,
                                                      "WS10m": 0.0
                                                  }.values()))
        BaseDataLoader.__init__(self,
                                file='../data/minutely/information.h5',
                                episode_length=episode_length)
        self.set_episode(episode)

        self.update_state()

    def step(self):
        self.update_reward_cache()
        self.step_time()
        self.update_state()

    def update_state(self):
        self.state = self.get_values(self.time).values
        super().update_state()

    def update_reward_cache(self):
        pass


if __name__ == "__main__":
    info = Information(0)
    import numpy as np

    print(np.min(info.episode["H_sun"]))

    info.step()
    print(info.reward_cache)
    print(info.state)
