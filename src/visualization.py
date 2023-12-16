import matplotlib.pyplot as plt


def visualize_scenario(callback):
    intensities = [info["observation"]["intensity"] for info in callback.infos]
    plt.plot(intensities, label="Intensity", alpha=0.8)
    consumptions = [info["observation"]["consumption"] for info in callback.infos]
    plt.plot(consumptions, label="Consumption", alpha=0.8)
    generations = [info["observation"]["generation"] for info in callback.infos]
    plt.plot(generations, label="Generation", alpha=0.8)

    plt.legend()
    plt.show()


def visualize_battery_behaviour(callback, env):
    state_of_charge = [info["observation"]["battery_state_of_charge"] for info in callback.infos]
    battery_actions = [info["action"][env.unwrapped.battery_slice] for info in callback.infos]

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


def visualize_reward(callback):
    rewards = [info["reward"] for info in callback.infos]
    produced_energy = [info["reward_cache"]["produced_energy"] for info in callback.infos]
    consumed_energy = [info["reward_cache"]["consumed_energy"] for info in callback.infos]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    color = 'tab:red'
    ax[0].set_xlabel('timestep')
    ax[0].set_ylabel('Reward', color=color)
    ax[0].plot(rewards, color=color, label="Reward")
    ax[0].tick_params(axis='y', labelcolor=color)
    ax[0].legend(loc="upper left")

    ax1 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis

    ax1.set_ylabel('Energy', color=color)  # we already handled the x-label with ax1
    ax1.plot(produced_energy, color="yellow", label="Produced Energy")
    ax1.plot(consumed_energy, color='black', label="Consumed Energy")
    ax1.tick_params(axis='y')
    ax1.legend(loc="upper right")

    ax[1].plot(produced_energy, label="Produced Energy", alpha=0.8)
    g_t = [info["reward_cache"]["G_t"] for info in callback.infos]
    ax[1].plot(g_t, label="G_t", alpha=0.8)
    if "D_t" in callback.infos[0]["reward_cache"].keys():
        d_t = [info["reward_cache"]["D_t"] for info in callback.infos]
        ax[1].plot(d_t, label="D_t", alpha=0.8)
    ax[1].legend()

    ax[2].plot(consumed_energy, label="Consumed Energy", alpha=0.8)
    l_t = [info["reward_cache"]["L_t"] for info in callback.infos]
    ax[2].plot(l_t, label="L_t", alpha=0.8)
    if "C_t" in callback.infos[0]["reward_cache"].keys():
        c_t = [info["reward_cache"]["C_t"] for info in callback.infos]
        ax[2].plot(c_t, label="C_t", alpha=0.8)
    if "s_{a,t}" in callback.infos[0]["reward_cache"].keys():
        s_at = [info["reward_cache"]["s_{a,t}"] for info in callback.infos]
        ax[2].plot(s_at, label="s_{a,t}", alpha=0.8)
    if "a_{tcl,t}" in callback.infos[0]["reward_cache"].keys():
        a_tcl_t = [info["reward_cache"]["a_{tcl,t}"]*info["reward_cache"]["L_{TCL}"] for info in callback.infos]
        ax[2].plot(a_tcl_t, label="a_{tcl,t}*L_{TCL}", alpha=0.8)
    ax[2].legend()

    plt.tight_layout()
    plt.show()