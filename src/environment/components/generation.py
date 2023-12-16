from src.environment.components.base.base_component import BaseComponent
from src.environment.components.base.data_loader import BaseDataLoader


class Generation(BaseComponent, BaseDataLoader):
    def __init__(self, episode: int = 0,
                 synthetic_data: bool = False,
                 episode_length: int = None,
                 normalise: bool = False
                 ):
        self.scaling = 100

        BaseComponent.__init__(self,
                               normalise=normalise,
                               max_state=440.0727161894718 * self.scaling,
                               min_state=-4.269742832803 * self.scaling)
        BaseDataLoader.__init__(self,
                                file='../data/minutely/generation.h5',
                                synthetic_data=synthetic_data,
                                episode_length=episode_length)
        self.set_episode(episode)

        self.counter = self.episode_length  # only for synthetic data

        self.update_state()

    def step(self):
        self.update_reward_cache()
        self.step_time()
        self.update_state()

    def update_state(self):
        if not self.synthetic_data:
            self.state = self.scaling * self.get_values(self.time)["AC"].values[0]  # in kW
            super().update_state()
        else:
            self.state = self.counter / self.episode_length
            self.counter -= 1

    def update_reward_cache(self):
        self.reward_cache["G_t"] = self.state


if __name__ == "__main__":
    gen = Generation(0)

    gen.step()
    print(gen.reward_cache)
    print(gen.state)
