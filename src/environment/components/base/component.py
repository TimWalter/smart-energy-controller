from abc import ABC, abstractmethod


class Component(ABC):
    def __init__(self):
        self.state = None
        self.reward_cache = {}

    @abstractmethod
    def update_state(self):
        pass

    @abstractmethod
    def update_reward_cache(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass
