import gymnasium as gym
import numpy as np


class SingleFamilyHome(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], "render_fps": 4}

    def __init__(self, planning_horizon: int, render_mode=None):
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = gym.spaces.Dict(
            {
                "controlled": gym.spaces.Dict(
                    {
                        "battery_state_of_charge": gym.spaces.Box(low=0, high=1, shape=(1,)),
                        "tcl_state_of_charge": gym.spaces.Box(low=0, high=1, shape=(1,)),
                        "controllable_appliances_planning_horizon": gym.spaces.MultiBinary(planning_horizon, seed=42),
                    }
                ),
                "uncontrolled": gym.spaces.Dict({
                    "generation": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "consumption": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "intensity": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                }),
                "informational": gym.spaces.Dict({
                    "time": gym.spaces.Text(min_length=26, max_length=26, charset="0123456789-:T."),
                    "solar_irradiation": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "solar_elevation": gym.spaces.Box(low=0, high=90, shape=(1,)),
                    "temperature": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                    "wind_speed": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
                }),
            }
        )

        self.action_space = gym.spaces.Dict({
            "battery": gym.spaces.Box(low=0, high=1, shape=(1,)),
            "tcl": gym.spaces.Discrete(2),
            "controllable_appliances": gym.spaces.MultiBinary(planning_horizon, seed=42),
        })

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
