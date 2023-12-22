from datetime import datetime
from typing import Callable

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.environment.single_family_home import SingleFamilyHome
from utils import LoggingCallback


def log(msg: str):
    print(f"[{datetime.now()}] {msg}")


def train(name: str, agent: Callable, policy: str, eval_epochs: int, train_epochs: int, check=True, logging=True):
    results = {}

    callback = None
    if logging:
        callback = LoggingCallback()

    log(f"Starting training for {name} with {agent.__name__} and {policy} policy")
    env = Monitor(SingleFamilyHome(), filename=f"./logs/{name}")

    if check:
        log("Checking environment")
        check_env(env)
        log("Environment checked")

    model = agent(policy, env, tensorboard_log=f"./tensorboard/")

    log("Evaluating untrained model")

    results["untrained_accumulated_reward"] = \
    evaluate_policy(model, env, n_eval_episodes=eval_epochs, callback=callback)[0]
    log("Untrained model evaluated")

    log("Starting training")
    model.learn(total_timesteps=train_epochs * 10080, callback=callback)
    log("Training finished")

    log("Evaluating trained model")
    results["trained_accumulated_reward"] = evaluate_policy(model, env, n_eval_episodes=eval_epochs, callback=callback)[0]
    log("Trained model evaluated")

    model.save(f"./models/{name}")

    if logging:
        callback.dump(f"./logs/{name}.json")

    return results


if __name__ == "__main__":
    from stable_baselines3 import SAC

    #results = train("SAC_1", SAC, "MultiInputPolicy", 1, 1)

    #print(results)

    env = Monitor(SingleFamilyHome(), filename=f"./logs/test")
    env.reset()
    env.step(env.action_space.sample())
