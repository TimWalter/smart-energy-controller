from gymnasium.envs.registration import register
from src.environment import SingleFamilyHome
from stable_baselines3 import DQN

register(
    id="SingleFamilyHome-v0",
    entry_point="environment.environment:SingleFamilyHome",
    max_episode_steps=10080,
    order_enforce=True,
)

model = DQN("MlpPolicy", "SingleFamilyHome-v0", verbose=1).learn(10080)

