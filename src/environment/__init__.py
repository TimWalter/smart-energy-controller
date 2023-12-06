from gymnasium.envs.registration import register
from src.environment.environment import SingleFamilyHome

register(
    id="SingleFamilyHome-v0",
    entry_point="environment.environment:SingleFamilyHome",
    max_episode_steps=10080,
    order_enforce=True,
)
