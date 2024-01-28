import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def overview(data_dict=None, path='./logs/minutely'):
    fig, ax = plt.subplots(2, 1, figsize=(20, 7 * 2))

    if data_dict is None:
        data_dict = load_data(path)
    reward_over_epochs(subplot=ax[0], path=path)
    best_rewards_by_category(data_dict, subplot=ax[1])

    plt.show()
    return data_dict


def add_data(data_dict, name, path="./logs/minutely"):
    data_dict[name] = pickle.load(open(path + "/" + name + ".pkl", 'rb'))
    return data_dict


def get_best_episodes(data_dict):
    best_episodes = {}
    for key, data in data_dict.items():
        best_episode = 0
        best_reward = -np.inf
        for i, episode in enumerate(data):
            if np.sum(episode["reward"]) > best_reward:
                best_reward = np.sum(episode["reward"])
                best_episode = i
        best_episodes[key] = best_episode
    return best_episodes


def load_data(path):
    data_dict = {}

    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        if os.path.isfile(item_path):
            data = pickle.load(open(item_path, 'rb'))
            if isinstance(data, list):
                data_dict[item.split(".")[0]] = data

    best_episodes = get_best_episodes(data_dict)
    data_dict = {k: v for k, v in sorted(data_dict.items(), key=lambda item: np.sum(item[1][best_episodes[item[0]]]["reward"]))}

    return data_dict


def load_results(path):
    files = os.listdir(path + "/results_only")
    results = {}
    for file in files:
        data = pickle.load(open(path + "/results_only/" + file, "rb"))
        if isinstance(data, list):
            results[file.split(".")[0]] = [d[0] for d in data]

    return results


def reward_over_epochs(figsize=(20, 7), subplot=None, path=None):
    results = load_results(path)

    cmap = plt.get_cmap('hsv')
    colors = cmap(np.linspace(0, 0.9, len(results)))

    if subplot is None:
        # Create a new figure with specified size
        plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        # Use the provided subplot
        ax = subplot

    for i, run in enumerate(results.keys()):
        if len(results[run]) > 1:
            ax.plot(results[run], label=run, color=colors[i], )
        else:
            ax.plot(results[run] * np.max([len(eps) for eps in results.values()]), label=run, color=colors[i], )

    ax.legend()
    ax.set_ylabel("Accumulated Reward")
    ax.set_xlabel("Epoch")
    ax.set_title("Accumulated Reward over Epochs")
    ax.set_ylim([-50000, -15000])
    ax.grid()

    if subplot is None:
        plt.show()

    return results


def best_rewards_by_category(data_dict, figsize=(20, 7), subplot=None):
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

    reward_types = ['given_reward', 'ess_reward', 'fdr_reward', 'tcl_reward', 'discomfort']
    rewards_sum = {reward_type: [] for reward_type in reward_types}

    best_episodes = get_best_episodes(data_dict)
    for key, data in data_dict.items():
        for reward_type in reward_types:
            rewards_sum[reward_type] += [np.sum(data[best_episodes[key]]["reward_info"].get(reward_type, np.zeros(167)))]

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
    ax.set_xticklabels([key + f" {best_episodes[key]}" for key in data_dict.keys()], rotation='vertical')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("Accumulated Reward by Category")

    if subplot is None:
        plt.show()


def deep_dive(data, episode=-1):
    actions = {}
    observations = {}
    if "ess_reward" in data[episode]["reward_info"].keys():
        actions["ess"] = np.array(data[episode]["action"])[:, 0]
        observations["ess"] = data[episode]["next_observation"]["energy_storage_system_charge"]
    if "fdr_reward" in data[episode]["reward_info"].keys():
        actions["fdr"] = np.mean(np.array(data[episode]["action"])[:, 1:-1], axis=1)
        observations["fdr"] = data[episode]["next_observation"]["flexible_demand_schedule"]
    if "tcl_reward" in data[episode]["reward_info"].keys():
        actions["tcl"] = np.array(data[episode]["action"])[:, -1]
        observations["tcl"] = data[episode]["next_observation"]["tcl_indoor_temperature"]

    # Extract the carbon intensity observation
    carbon_intensity = np.array(data[episode]["next_observation"]["carbon_intensity"])[:,:, -1]

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
        ax2.plot(carbon_intensity, color='tab:red', label="Carbon Intensity", linewidth=5, alpha=0.2)
        ax2.tick_params(axis='y', labelcolor='tab:red')

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Move the third axis to the right
        ax3.set_ylabel('Observation', color='tab:orange')
        ax3.plot(observations[action], color='tab:orange', label="Observation")
        ax3.tick_params(axis='y', labelcolor='tab:orange')

    plt.tight_layout()
    plt.show()


def reward_comparison(data1, data2, episode=[-1, -1], cumulative=[False, False, False], diff=False):
    rewards_1 = {}
    rewards_2 = {}
    if "ess_reward" in data1[episode[0]]["reward_info"].keys():
        if cumulative[0]:
            rewards_1["ess"] = np.cumsum(np.array(data1[episode[0]]["reward_info"]["ess_reward"]))
            rewards_2["ess"] = np.cumsum(np.array(data2[episode[1]]["reward_info"]["ess_reward"]))
        else:
            rewards_1["ess"] = np.array(data1[episode[0]]["reward_info"]["ess_reward"])
            rewards_2["ess"] = np.array(data2[episode[1]]["reward_info"]["ess_reward"])
    if "fdr_reward" in data1[episode[0]]["reward_info"].keys():
        if cumulative[1]:
            rewards_1["fdr"] = np.cumsum(np.array(data1[episode[0]]["reward_info"]["fdr_reward"]))
            rewards_2["fdr"] = np.cumsum(np.array(data2[episode[1]]["reward_info"]["fdr_reward"]))
        else:
            rewards_1["fdr"] = np.array(data1[episode[0]]["reward_info"]["fdr_reward"])
            rewards_2["fdr"] = np.array(data2[episode[1]]["reward_info"]["fdr_reward"])
    if "tcl_reward" in data1[episode[0]]["reward_info"].keys():
        if cumulative[-1]:
            rewards_1["tcl"] = np.cumsum(np.array(data1[episode[0]]["reward_info"]["tcl_reward"]))
            rewards_2["tcl"] = np.cumsum(np.array(data2[episode[1]]["reward_info"]["tcl_reward"]))
            rewards_1["discomfort"] = np.cumsum(np.array(data1[episode[0]]["reward_info"]["discomfort"]))
            rewards_2["discomfort"] = np.cumsum(np.array(data2[episode[1]]["reward_info"]["discomfort"]))
        else:
            rewards_1["tcl"] = np.array(data1[episode[0]]["reward_info"]["tcl_reward"])
            rewards_2["tcl"] = np.array(data2[episode[1]]["reward_info"]["tcl_reward"])
            rewards_1["discomfort"] = np.array(data1[episode[0]]["reward_info"]["discomfort"])
            rewards_2["discomfort"] = np.array(data2[episode[1]]["reward_info"]["discomfort"])

    # Extract the carbon intensity observation
    carbon_intensity = np.array(data1[episode[0]]["next_observation"]["carbon_intensity"])

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
