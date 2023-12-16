import numpy as np

from src.environment.components.base.base_component import BaseComponent
from src.environment.components.base.data_loader import BaseDataLoader


class Generation(BaseComponent, BaseDataLoader):
    def __init__(self, episode: int = 0, synthetic_data: bool = False, episode_length: int = None):
        BaseDataLoader.__init__(self, file='../data/minutely/generation.h5', synthetic_data=synthetic_data,
                                episode_length=episode_length)
        self.set_episode(episode)

        self.update_state()
        BaseComponent.__init__(self, initial_state=self.state)

    def step(self):
        self.update_reward_cache()
        self.step_time()
        self.update_state()

    def update_state(self):
        if not self.synthetic_data:
            self.state = 100 * np.array(self.get_values(self.time)["AC"].values, dtype=np.float32)  # in kW
        else:
            self.state = np.array([100], dtype=np.float32)

    def update_reward_cache(self):
        self.reward_cache["G_t"] = self.state


if __name__ == "__main__":
    gen = Generation(0)

    gen.step()
    print(gen.reward_cache)
    print(gen.state)
