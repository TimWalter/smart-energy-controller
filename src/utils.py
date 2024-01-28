import os
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class LoggedEvalCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.infos = []

    def _on_step(self) -> bool:
        self.infos.append(self.locals["infos"][0])
        return True

    def __call__(self, locals_, globals_):
        self.locals = locals_
        self.globals = globals_
        self._on_step()


def evaluate_policy_logged(model, env, n_eval_episodes, results, infos):
    for _ in range(n_eval_episodes):
        eval_callback = LoggedEvalCallback()
        results += [evaluate_policy(model, env, n_eval_episodes=1, callback=eval_callback, warn=False)]
        infos += [eval_callback.infos]


class TrainCallback(BaseCallback):
    def __init__(self, results, infos, best_model_save_path, eval_freq, eval_env, n_eval_episodes, verbose=0):
        super().__init__(verbose)
        self.results = results
        self.infos = infos
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.best_model_save_path = best_model_save_path
        self.current_eval = 0
        os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            evaluate_policy_logged(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                                   results=self.results,
                                   infos=self.infos)
            print(f"[{datetime.now()}] Reward {self.current_eval}: {self.results[-self.n_eval_episodes:]}")
            self.current_eval += 1
            if self.results[-1][0] > self.best_mean_reward:
                self.best_mean_reward = self.results[-1][0]
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
        return True
