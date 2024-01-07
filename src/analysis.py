import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

episode_length = 167


def overview(data_dict=None, path='./logs/minutely'):
    fig, ax = plt.subplots(2, 1, figsize=(20, 7 * 2))

    if data_dict is None:
        data_dict = load_data(path)
    reward_over_epochs(subplot=ax[0], path=path)
    final_rewards_by_category(data_dict, subplot=ax[1])

    plt.show()
    return data_dict


def add_data(data_dict, name, path="./logs/minutely"):
    data_dict[name] = pickle.load(open(path + "/" + name + ".pkl", 'rb'))
    return data_dict


def load_data(path):
    data_dict = {}

    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        if os.path.isfile(item_path):
            data_dict[item.split(".")[0]] = pickle.load(open(item_path, 'rb'))

    data_dict = {k: v for k, v in
                 sorted(data_dict.items(), key=lambda item: np.sum(item[1]["reward"][-episode_length:]))}

    return data_dict


def load_results(path):
    files = os.listdir(path + "/results_only")
    runs = {
        file.split(".")[0]: np.array(
            [el if isinstance(el, float) else el[0] for el in
             pickle.load(open(path + "/results_only/" + file, "rb")).values()])
        for file in files
    }
    return runs


def reward_over_epochs(figsize=(20, 7), subplot=None, path=None):
    files = os.listdir(path + "/results_only")
    runs = {
        file.split(".")[0]: np.array(
            [el if isinstance(el, float) else el[0] for el in
             pickle.load(open(path + "/results_only/" + file, "rb")).values()])
        for file in files
    }

    cmap = plt.get_cmap('hsv')
    colors = cmap(np.linspace(0, 0.9, len(runs)))

    if subplot is None:
        # Create a new figure with specified size
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        # Use the provided subplot
        ax = subplot

    for i, run in enumerate(runs.keys()):
        if len(runs[run]) > 1:
            ax.plot(runs[run], label=run, color=colors[i], )
        else:
            ax.plot(runs[run], label=run, marker='o', color=colors[i], )
    ax.legend()
    ax.set_ylabel("Accumulated Reward")
    ax.set_xlabel("Epoch")
    ax.set_title("Accumulated Reward over Epochs")
    # ax.set_ylim(bottom=0, top=1000)

    if subplot is None:
        plt.show()

    return runs


def final_rewards_by_category(data_dict, figsize=(20, 7), subplot=None):
    def get_rewards_sum(data, reward_type):
        rewards = np.array(data["cache"].get(reward_type, np.zeros(episode_length))[-episode_length:])
        return np.sum(rewards)

    def plot_rewards(ax, x, rewards, bottom, label, color, is_positive):
        filtered_rewards = [max(0, reward) if is_positive else min(0, reward) for reward in rewards]
        bars = ax.bar(x, filtered_rewards, bottom=bottom, color=color, label=label if is_positive else None)
        bottom += filtered_rewards
        for j, rect in enumerate(bars):
            height = rect.get_height()
            if height != 0:
                ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_y() + height / 2.0, str(round(height, 2)),
                        ha='center', va='center')
        return bottom

    reward_types = ['given_reward', 'battery_reward', 'fdr_reward', 'tcl_reward', 'discomfort']
    rewards_sum = {reward_type: [] for reward_type in reward_types}

    for key, data in data_dict.items():
        for reward_type in reward_types:
            rewards_sum[reward_type].append(get_rewards_sum(data, reward_type))

    if subplot is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        ax = subplot

    x = np.arange(len(data_dict))
    bottom_positive = np.zeros(len(data_dict))
    bottom_negative = np.zeros(len(data_dict))

    colors = plt.get_cmap('hsv')(np.linspace(0, 0.9, len(rewards_sum)))

    for i, (label, rewards) in enumerate(rewards_sum.items()):
        bottom_positive = plot_rewards(ax, x, rewards, bottom_positive, label, colors[i], is_positive=True)
        bottom_negative = plot_rewards(ax, x, rewards, bottom_negative, label, colors[i], is_positive=False)

    ax.set_xticks(x)
    ax.set_xticklabels(data_dict.keys(), rotation='vertical')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("Accumulated Reward by Category")

    if subplot is None:
        plt.show()


def deep_dive(data, episode=-1):
    data["action"] = np.array(data["action"])

    eps_slice = slice(episode_length * episode, episode_length * (episode + 1) if episode != -1 else None)

    actions = {}
    observations = {}
    if "battery_reward" in data["cache"].keys():
        actions["ess"] = data["action"][eps_slice, 0]
        observations["ess"] = data["next_observation"]["energy_storage_system_charge"][eps_slice]
    if "fdr_reward" in data["cache"].keys():
        actions["fdr"] = data["action"][eps_slice, 1 if "battery_reward" in data["cache"].keys() else 0]
        observations["fdr"] = data["next_observation"]["flexible_demand_schedule"][eps_slice]
    if "tcl_reward" in data["cache"].keys():
        actions["tcl"] = data["action"][eps_slice, -1]
        observations["tcl"] = data["next_observation"]["tcl_indoor_temperature"][eps_slice]

    # Extract the carbon intensity observation
    carbon_intensity = np.array(data["next_observation"]["carbon_intensity"][-episode_length:])

    # Create a subplot for each action
    fig, axs = plt.subplots(len(actions), 1, figsize=(20, 7 * len(actions)))

    for i, (action, action_data) in enumerate(actions.items()):
        ax1 = axs[i] if len(actions) > 1 else axs
        ax1.set_xlabel('timestep')
        ax1.set_ylabel('action', color='tab:blue')
        ax1.plot(action_data, color='tab:blue', label=action)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_title(action)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Carbon Intensity', color='tab:red')
        ax2.plot(carbon_intensity, color='tab:red', label="Carbon Intensity")
        ax2.tick_params(axis='y', labelcolor='tab:red')

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Move the third axis to the right
        ax3.set_ylabel('Observation', color='tab:green')
        ax3.plot(observations[action], color='tab:green', label="Observation")
        ax3.tick_params(axis='y', labelcolor='tab:green')

    plt.tight_layout()
    plt.show()


def reward_comparison(data1, data2, episode=[-1, -1], cumulative=[False, False, False], diff=False):
    eps_slice1 = slice(episode_length * episode[0], episode_length * (episode[0] + 1) if episode[0] != -1 else None)
    eps_slice2 = slice(episode_length * episode[1], episode_length * (episode[1] + 1) if episode[1] != -1 else None)

    rewards_1 = {}
    rewards_2 = {}
    if "battery_reward" in data1["cache"].keys():
        if cumulative[0]:
            rewards_1["ess"] = np.cumsum(np.array(data1["cache"]["battery_reward"][eps_slice1]))
            rewards_2["ess"] = np.cumsum(np.array(data2["cache"]["battery_reward"][eps_slice2]))
        else:
            rewards_1["ess"] = np.array(data1["cache"]["battery_reward"][eps_slice1])
            rewards_2["ess"] = np.array(data2["cache"]["battery_reward"][eps_slice2])
    if "fdr_reward" in data1["cache"].keys():
        if cumulative[1]:
            rewards_1["fdr"] = np.cumsum(np.array(data1["cache"]["fdr_reward"][eps_slice1]))
            rewards_2["fdr"] = np.cumsum(np.array(data2["cache"]["fdr_reward"][eps_slice2]))
        else:
            rewards_1["fdr"] = np.array(data1["cache"]["fdr_reward"][eps_slice1])
            rewards_2["fdr"] = np.array(data2["cache"]["fdr_reward"][eps_slice2])
    if "tcl_reward" in data1["cache"].keys():
        if cumulative[-1]:
            rewards_1["tcl"] = np.cumsum(np.array(data1["cache"]["tcl_reward"][eps_slice1]))
            rewards_2["tcl"] = np.cumsum(np.array(data2["cache"]["tcl_reward"][eps_slice2]))
            rewards_1["discomfort"] = np.cumsum(np.array(data1["cache"]["discomfort"][eps_slice1]))
            rewards_2["discomfort"] = np.cumsum(np.array(data2["cache"]["discomfort"][eps_slice2]))
        else:
            rewards_1["tcl"] = np.array(data1["cache"]["tcl_reward"][eps_slice1])
            rewards_2["tcl"] = np.array(data2["cache"]["tcl_reward"][eps_slice2])
            rewards_1["discomfort"] = np.array(data1["cache"]["discomfort"][eps_slice1])
            rewards_2["discomfort"] = np.array(data2["cache"]["discomfort"][eps_slice2])

    # Extract the carbon intensity observation
    carbon_intensity = np.array(data1["next_observation"]["carbon_intensity"][eps_slice1])

    # Create a subplot for each action
    fig, axs = plt.subplots(len(rewards_1), 1, figsize=(20, 7 * len(rewards_1)))

    for i, (reward, reward_data) in enumerate(rewards_1.items()):
        ax1 = axs[i] if len(rewards_1) > 1 else axs
        ax1.set_xlabel('timestep')
        ax1.set_ylabel('reward', color='tab:blue')
        ax1.plot(reward_data, color='tab:blue', label="1")
        ax1.plot(rewards_2[reward], color='tab:orange', label="2")
        if diff:
            ax1.plot(rewards_1[reward] - rewards_2[reward], color='tab:green', label="1-2", alpha=0.5)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_title(reward)
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.set_ylabel('Carbon Intensity', color='tab:red')
        ax2.plot(carbon_intensity, color='tab:red', label="Carbon Intensity")
        ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.tight_layout()
    plt.show()
