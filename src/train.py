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
    env = Monitor(SingleFamilyHome())

    if check:
        log("Checking environment")
        check_env(env)
        log("Environment checked")

    model = agent(policy, env)

    log("Evaluating untrained model")
    #env.unwrapped.eval()
    results["untrained_accumulated_reward"] = \
        evaluate_policy(model, env, n_eval_episodes=eval_epochs, callback=callback)
    log("Untrained model evaluated")
    log(f"Untrained accumulated reward: {results['untrained_accumulated_reward']}")

    log("Starting training")
    for epoch in range(train_epochs):
        log(f"Training epoch {epoch + 1}/{train_epochs}")
        #env.unwrapped.train()
        model.learn(total_timesteps=10080, callback=callback)

        #env.unwrapped.eval()
        results[f"epoch_{epoch}_accumulated_reward"] = \
            evaluate_policy(model, env, n_eval_episodes=eval_epochs, callback=callback)
        log(f"Epoch {epoch + 1}/{train_epochs} finished")
        log(f"Accumulated reward: {results[f'epoch_{epoch}_accumulated_reward']}")
    log("Training finished")

    model.save(f"./models/{name}")

    if logging:
        callback.dump(f"./logs/{name}.pkl")

    return results


if __name__ == "__main__":
    import pickle
    from stable_baselines3 import SAC, PPO
    from baselines.idle import Idle
    from baselines.single_threshold import SingleThreshold

    name = "single_threshold_develop_discrete"
    results = train(name, SingleThreshold, "MultiInputPolicy", 1, 0, check=False)
    pickle.dump(results, open(f"./logs/results_only/{name}", "wb"))
