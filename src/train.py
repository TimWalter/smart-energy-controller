import pickle
import os
from datetime import datetime
from typing import Callable

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from baselines.idle import Idle
from baselines.mirror import Mirror
from baselines.single_threshold import SingleThreshold
from src.environment.stripped_single_family_home import SingleFamilyHome
from utils import TrainCallback, evaluate_policy_logged


def log(msg: str):
    print(f"[{datetime.now()}] {msg}")


def train(
        path: str,
        name: str,
        agent: Callable,
        policy: str,
        eval_epochs: int,
        train_epochs: int,
        check: bool,
        config: str):
    results = []
    infos = []

    eval_env = Monitor(SingleFamilyHome(config=config))
    # eval_env.unwrapped.eval()
    train_env = SingleFamilyHome(config=config)
    # train_env.train()
    test_env = Monitor(SingleFamilyHome(config=config))
    # test_env.unwrapped.test()

    if check:
        log("Checking environment")
        check_env(eval_env)
        log("Environment checked")

    episode_length = 10079 if eval_env.unwrapped.resolution == "minutely" else 167

    log(f"Starting training for {name} with {agent.__name__} and {policy} policy")

    if agent == PPO:
        model = agent(policy, train_env, seed=15, n_steps=167, batch_size=167, learning_rate=1e-6, vf_coef=5e-2)
    else:
        model = agent(policy, train_env, seed=15, learning_rate=1e-6)

    evaluate_policy_logged(model, eval_env, n_eval_episodes=eval_epochs, results=results, infos=infos)
    log(f"Untrained accumulated reward: {results[-1]}")

    if train_epochs:
        log("Starting training")
        train_callback = TrainCallback(results, infos, f"./models/{path}/{name}", episode_length, eval_env, eval_epochs)
        model.learn(total_timesteps=episode_length * train_epochs, callback=train_callback)
        log("Training finished")

        evaluate_policy_logged(model, eval_env, n_eval_episodes=eval_epochs, results=results, infos=infos)
        log(f"Trained Accumulated reward: {results[-1]}")

    def group_infos(infos):
        return [
            {
                key: {key_inner: [timestep[key][key_inner] for timestep in episode] for key_inner in
                      episode[0][key].keys()}
                if isinstance(episode[0][key], dict)
                else [timestep[key] for timestep in episode]
                for key in episode[0].keys()
            }
            for episode in infos
        ]

    info_path = f"./logs/{path}"
    os.makedirs(info_path, exist_ok=True)
    pickle.dump(group_infos(infos), open(f"{info_path}/{name}.pkl", "wb"))

    results_path = f"./logs/{path}/results_only"
    os.makedirs(results_path, exist_ok=True)
    pickle.dump(results, open(f"{results_path}/{name}.pkl", "wb"))


if __name__ == "__main__":

    agents = {
        "ppo": PPO,
        "sac": SAC,
        "idle": Idle,
        "single-threshold": SingleThreshold,
        "mirror": Mirror,
    }

    folder_path = "environment/configs/config_"
    for path in ["ess"]:
        for name, train_epochs in zip(["idle", "single-threshold", "mirror", "ppo_test", "sac_test"], [0,0,0,10,10]):
            train(
                path,
                name,
                agents[name.split("_")[0]],
                "MultiInputPolicy",
                1,
                train_epochs,
                False,
                f"environment/configs/config_{path}.json")
