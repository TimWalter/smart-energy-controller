import pickle

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.infos = []

    def dump(self, path: str):
        with open(path, "wb") as f:
            self.infos = {
                key: {key_inner: [info[key][key_inner] for info in self.infos] for key_inner in self.infos[0][key].keys()}
                if isinstance(self.infos[0][key], dict)
                else [info[key] for info in self.infos]
                for key in self.infos[0].keys()
            }

            pickle.dump(self.infos, f)
        self.infos = []

    def _on_step(self) -> bool:
        self.infos.append(self.locals["infos"][0])
        return True

    def __call__(self, locals_, globals_):
        self.locals = locals_
        self.globals = globals_
        self._on_step()


def visualize_scenario(callback, env):
    intensities = [info["next_observation"]["carbon_intensity"] for info in callback.infos]
    consumptions = [info["next_observation"]["household_energy_demand"] for info in callback.infos]
    generations = [info["next_observation"]["rooftop_solar_generation"] for info in callback.infos]

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('timestep')
    ax1.set_ylabel('Carbon Intensity', color=color)
    ax1.plot(intensities, color=color, label="Carbon Intensity", alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    ax2.set_ylabel('Energy', color="red")  # we already handled the x-label with ax1
    ax2.plot(np.array(generations) - np.array(consumptions), color="red", label="Generation-Consumption", alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()


def visualize_battery_behaviour(callback, env):
    state_of_charge = [info["next_observation"]["battery_state_of_charge"] for info in callback.infos]
    battery_actions = [info["action"][env.action_slice["energy_storage_system"]] for info in callback.infos]

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('timestep')
    ax1.set_ylabel('State of Charge', color=color)
    ax1.plot(state_of_charge, color=color, label="State of Charge")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Battery Action', color=color)  # we already handled the x-label with ax1
    ax2.plot(battery_actions, color=color, label="Battery Action")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def visualize_tcl_behaviour(callback, env):
    tcl_temperatures = [info["next_observation"]["tcl_state_of_charge"] for info in callback.infos]
    tcl_actions = [info["action"][env.action_slice["thermostatically_controlled_load"]] for info in callback.infos]

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('timestep')
    ax1.set_ylabel('State of Charge', color=color)
    ax1.plot(tcl_temperatures, color=color, label="State of Charge")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('TCL Action', color=color)  # we already handled the x-label with ax1
    ax2.plot(tcl_actions, color=color, label="TCL Action")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def visualize_reward(callback, env):
    rewards = [info["reward"] for info in callback.infos]
    household_energy_demand = [info["next_observation"]["household_energy_demand"] for info in callback.infos]
    rooftop_solar_generation = [info["next_observation"]["rooftop_solar_generation"] for info in callback.infos]
    if "energy_storage_system" in env.action_slice.keys():
        battery_actions = [info["action"][env.action_slice["energy_storage_system"]] for info in callback.infos]

    if "flexible_demand_response" in env.action_slice.keys():
        fdr_actions = [info["action"][env.action_slice["flexible_demand_response"]] for info in callback.infos]

    if "thermostatically_controlled_load" in env.action_slice.keys():
        tcl_actions = [info["action"][env.action_slice["thermostatically_controlled_load"]] for info in callback.infos]

    fig, ax1 = plt.subplots()
    color = 'tab:green'
    ax1.set_xlabel('timestep')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(rewards, color=color, label="Reward")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    if "energy_storage_system" in env.action_slice.keys():
        ax2.plot(battery_actions, color="blue", label="Battery Action")
    if "flexible_demand_response" in env.action_slice.keys():
        ax2.plot(fdr_actions, color="red", label="FDR Action")
    if "thermostatically_controlled_load" in env.action_slice.keys():
        ax2.plot(tcl_actions, color="purple", label="TCL Action")
    ax2.legend(loc="upper right")

    ax3 = ax1.twinx()
    ax3.set_ylabel('Energy', color="orange")  # we already handled the x-label with ax1
    ax3.tick_params(axis='y', labelcolor="orange")
    ax3.plot(np.array(rooftop_solar_generation) - np.array(household_energy_demand), color="orange",
             label="Generation-Consumption", alpha=0.8)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
