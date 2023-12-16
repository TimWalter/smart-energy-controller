from abc import ABC, abstractmethod

import numpy as np


class BaseComponent(ABC):
    def __init__(self, normalise: bool, max_state: float | np.ndarray, min_state: float | np.ndarray):
        self.state = None
        self.reward_cache = {}

        self.normalise = normalise
        self.max_state = max_state
        self.min_state = min_state

    def update_state(self):
        if self.normalise:
            self.state = (self.state - self.min_state) / (self.max_state - self.min_state)

    @abstractmethod
    def update_reward_cache(self):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass
