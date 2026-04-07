import numpy as np
import gymnasium as gym


class OneHotObservationWrapper(gym.ObservationWrapper):
    """
    Convert a Discrete observation (e.g., FrozenLake state id) into a one-hot vector.

    Output observation_space becomes:
      Box(low=0, high=1, shape=(n_states,), dtype=float32)
    """

    def __init__(self, env, n_states: int | None = None, dtype=np.float32):
        super().__init__(env)
        if not isinstance(self.observation_space, gym.spaces.Discrete):
            raise TypeError(
                f"OneHotObservationWrapper expects Discrete obs space, got {type(self.observation_space)}"
            )

        self.n_states = int(n_states) if n_states is not None else int(self.observation_space.n)
        self.dtype = dtype

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_states,),
            dtype=self.dtype,
        )

    def observation(self, observation):
        one_hot = np.zeros(self.n_states, dtype=self.dtype)
        one_hot[int(observation)] = 1.0
        return one_hot

