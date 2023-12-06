from abc import ABC, abstractmethod


class BaseComponent(ABC):
    def __init__(self, initial_state):
        self.state = initial_state
        self.reward_cache = {}

    @abstractmethod
    def update_state(self, *args, **kwargs):
        pass

    @abstractmethod
    def update_reward_cache(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass
