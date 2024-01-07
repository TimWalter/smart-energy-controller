from datetime import datetime
from typing import Callable

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.environment.single_family_home_full import SingleFamilyHome
from utils import LoggingCallback, gather_experiences, TrainCallback, NormalizeDictObservation
from stable_baselines3 import SAC, PPO
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
        config: str,
        warm_start: bool):

    results = {}
    infos = []

    eval_env = Monitor(SingleFamilyHome(config=config))
    #eval_env.unwrapped.eval()
    train_env = SingleFamilyHome(config=config)
    #train_env.train()
    test_env = Monitor(SingleFamilyHome(config=config))
    #test_env.unwrapped.test()

    if check:
        log("Checking environment")
        check_env(eval_env)
        log("Environment checked")

    episode_length = 10079 if eval_env.unwrapped.resolution == "minutely" else 167

    log(f"Starting training for {name} with {agent.__name__} and {policy} policy")

    if agent == PPO:
        model = agent(policy, train_env, seed=15, target_kl=0.5, n_steps=167, batch_size=167)
    else:
        model = agent(policy, train_env, seed=15)
    if warm_start and agent == SAC:
        warmup_agent = SingleThreshold(None, eval_env)
        initial_experiences = gather_experiences(warmup_agent, eval_env, episode_length)
        for experience in initial_experiences:
            model.replay_buffer.add(*experience)

    eval_callback = LoggingCallback()
    results["untrained_accumulated_reward"] = evaluate_policy(model, eval_env, n_eval_episodes=eval_epochs,
                                                              callback=eval_callback)

    log(f"Untrained accumulated reward: {results['untrained_accumulated_reward']}")

    if train_epochs:
        log("Starting training")
        train_callback = TrainCallback(eval_env, best_model_save_path=f"./models/{path}/{name}",
                                       eval_freq=episode_length, n_eval_episodes=eval_epochs)
        model.learn(total_timesteps=episode_length * train_epochs, callback=train_callback)
        infos += train_callback.infos
        results.update(train_callback.results)
        log("Training finished")

        results[f"trained_accumulated_reward"] = evaluate_policy(model, eval_env, n_eval_episodes=eval_epochs,
                                                                 callback=eval_callback)
        log(f"Trained Accumulated reward: {results[f'trained_accumulated_reward']}")

    results[f"test"] = evaluate_policy(model, test_env, n_eval_episodes=4, callback=eval_callback)

    infos += eval_callback.infos

    def strip_infos(infos):
        return {
            key: {key_inner: [info[key][key_inner] for info in infos] for key_inner in
                  infos[0][key].keys()}
            if isinstance(infos[0][key], dict)
            else [info[key] for info in infos]
            for key in infos[0].keys()
        }


    pickle.dump(strip_infos(infos), open(f"./logs/{path}/{name}.pkl", "wb"))
    pickle.dump(results, open(f"./logs/{path}/results_only/{name}", "wb"))


if __name__ == "__main__":
    import pickle

    from baselines.single_threshold import SingleThreshold
    from baselines.idle import Idle

    agents = {
        "ppo": PPO,
        "sac": SAC,
        "idle": Idle,
        "single-threshold": SingleThreshold,
    }

    folder_path = "environment/configs/config_"
    for path in ["hourly"]:
        for name, train_epochs in zip(["sac_curious"],[1000]):
            train(
                path,
                name,
                agents[name.split("_")[0]],
                "MultiInputPolicy",
                1,
                train_epochs,
                False,
                f"environment/configs/config_{path}.json",
                True)
