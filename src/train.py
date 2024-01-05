from datetime import datetime
from typing import Callable

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.environment.single_family_home import SingleFamilyHome
from utils import LoggingCallback, gather_experiences, CustomEvalCallback


def log(msg: str):
    print(f"[{datetime.now()}] {msg}")


def learning_rate_schedule(progress: float) -> float:
    start_lr = 0.015  # starting learning rate
    end_lr = 0.0001  # ending learning rate
    return start_lr - progress * (start_lr - end_lr)


def train(
        path: str,
        name: str,
        agent: Callable,
        policy: str,
        eval_epochs: int,
        train_epochs: int,
        check: bool,
        logging: bool,
        config: str,
        warm_start: bool):
    results = {}
    eval_env = Monitor(SingleFamilyHome(config=config))

    if check:
        log("Checking environment")
        check_env(eval_env)
        log("Environment checked")

    episode_length = 10079 if eval_env.unwrapped.resolution == "minutely" else 167

    logging_callback = None
    if logging:
        logging_callback = LoggingCallback()

    log(f"Starting training for {name} with {agent.__name__} and {policy} policy")

    model = agent(policy, SingleFamilyHome(config=config), seed=42)

    if warm_start:
        warmup_agent = SingleThreshold(None, eval_env)
        initial_experiences = gather_experiences(warmup_agent, eval_env, episode_length)
        for experience in initial_experiences:
            model.replay_buffer.add(*experience)

    log("Evaluating untrained model")
    # env.unwrapped.eval()
    results["untrained_accumulated_reward"] = evaluate_policy(model, eval_env, n_eval_episodes=eval_epochs,
                                                              callback=logging_callback)
    log("Untrained model evaluated")
    log(f"Untrained accumulated reward: {results['untrained_accumulated_reward']}")

    if train_epochs:
        log("Starting training")
        # env.unwrapped.train()
        eval_callback = CustomEvalCallback(eval_env, best_model_save_path=f"./models/{path}/{name}",
                                           eval_freq=episode_length, n_eval_episodes=eval_epochs)
        model.learn(total_timesteps=episode_length * train_epochs, callback=eval_callback)
        # env.unwrapped.eval()
        results.update(eval_callback.results)
        log("Training finished")

        results[f"trained_accumulated_reward"] = evaluate_policy(model, eval_env, n_eval_episodes=eval_epochs,
                                                                 callback=logging_callback)
        log(f"Trained Accumulated reward: {results[f'trained_accumulated_reward']}")

    if logging:
        logging_callback.dump(f"./logs/{path}/{name}.pkl")

    pickle.dump(results, open(f"./logs/{path}/results_only/{name}", "wb"))


if __name__ == "__main__":
    import pickle
    from stable_baselines3 import SAC, PPO
    from baselines.single_threshold import SingleThreshold
    from baselines.idle import Idle

    agents = {
        "ppo": PPO,
        "sac": SAC,
        "idle": Idle,
        "single_threshold": SingleThreshold
    }

    folder_path = "environment/configs/config_"
    for path in ["ess", "fdr", "tcl", "hourly"]:
        for name, train_epochs in zip(["idle", "single_threshold", "ppo", "sac"],
                                      [0, 0, 10, 10]):
            train(
                path,
                name,
                agents[name.split("_")[0]],
                "MultiInputPolicy",
                1,
                train_epochs,
                False,
                True,
                f"environment/configs/config_{path}.json",
                False)