import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

_MAPS_PATH = Path(__file__).resolve().parent.parent / "maps.json"


def load_map(size: int):
    """Load grid layout from project-root maps.json (keys \"4\", \"8\", \"16\", ...)."""
    with open(_MAPS_PATH, encoding="utf-8") as f:
        return json.load(f)[str(size)]


def max_episode_steps_for_map_size(size: int) -> int:
    """Larger maps need a higher step limit so episodes do not mostly end in timeout."""
    return {4: 100, 8: 200, 16: 500}.get(size, max(100, size * size * 2))


def plot_subdir_for_map_size(map_size):
    """
    Subfolder name under dqn_plots/ (automatic from MAP_SIZE).
    4 -> "4x4", 8 -> "8x8", 16 -> "16x16"; None -> Gym default layout folder name.
    """
    if map_size is None:
        return "gym_default"
    return f"{int(map_size)}x{int(map_size)}"


class _TrainLogger(BaseCallback):
    """Episode + TD stats for DQN_plots.save_training_visualization."""

    def __init__(self, max_episode_steps=100, td_eval_freq=500, td_batch_size=64, success_window=50):
        super().__init__(0)
        self.max_episode_steps = max_episode_steps
        self.td_eval_freq = td_eval_freq
        self.td_batch_size = td_batch_size
        self.success_window = success_window
        self.episode_rewards = []
        self.episode_success = []
        self.episode_end_steps = []
        self.outcomes = []
        self.episode_times = []
        self.steps_per_episode = []
        self.avg_reward_steps = []
        self.avg_rewards = []
        self.success_rate_episodes = []
        self.success_rates = []
        self.td_steps = []
        self.td_errors = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            r, L, t = info["episode"]["r"], int(info["episode"]["l"]), float(info["episode"]["t"])
            to = bool(info.get("TimeLimit.truncated", False))
            if not to and r <= 0 and L >= self.max_episode_steps:
                to = True
            self.episode_rewards.append(r)
            ok = self._episode_success_from_info(info, r)
            self.episode_success.append(1 if ok else 0)
            self.episode_end_steps.append(self.num_timesteps)
            self.episode_times.append(t)
            self.steps_per_episode.append(L)
            self.outcomes.append(self._episode_outcome_from_info(info, r, to))
            lo = max(0, len(self.episode_rewards) - self.success_window)
            self.avg_reward_steps.append(self.num_timesteps)
            self.avg_rewards.append(float(np.mean(self.episode_rewards[lo:])))
            self.success_rate_episodes.append(len(self.episode_rewards))
            self.success_rates.append(float(np.mean(self.episode_success[lo:])))

        if self.num_timesteps % self.td_eval_freq == 0:
            rb = self.model.replay_buffer
            if rb is not None and rb.size() >= self.td_batch_size:
                s = rb.sample(self.td_batch_size, env=self.model._vec_normalize_env)
                with torch.no_grad():
                    q = self.model.q_net(s.observations)
                    cur = torch.gather(q, 1, s.actions.long())
                    nq = self.model.q_net_target(s.next_observations)
                    tgt = s.rewards + (1 - s.dones) * self.model.gamma * nq.max(1, keepdim=True)[0]
                    self.td_steps.append(self.num_timesteps)
                    self.td_errors.append(torch.abs(tgt - cur).mean().item())

        return True

    @staticmethod
    def _episode_success_from_info(info: dict, ep_return: float) -> bool:
        if "is_goal" in info:
            return bool(info["is_goal"])
        return ep_return > 0

    @staticmethod
    def _episode_outcome_from_info(info: dict, ep_return: float, timeout: bool) -> str:
        if _TrainLogger._episode_success_from_info(info, ep_return):
            return "success"
        if timeout:
            return "timeout"
        return "hole"


def make_env(
    render_mode=None,
    map_desc=None,
    is_slippery=False,
    max_episode_steps=None,
    reward_shaping: bool = False,
    one_hot: bool = False,
):
    """
    If map_desc is None, use Gym's default FrozenLake 4x4 layout.
    Otherwise pass desc=map_desc (e.g. from load_map(8)).
    Reward shaping (if enabled) wraps FrozenLake before one-hot so states stay Discrete.
    """
    kwargs = {"is_slippery": is_slippery, "render_mode": render_mode}
    if map_desc is not None:
        kwargs["desc"] = map_desc
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    env = gym.make("FrozenLake-v1", **kwargs)
    if reward_shaping:
        try:
            from DQN.reward_shaping_wrapper import FrozenLakeRewardShapingWrapper
        except ImportError:
            from reward_shaping_wrapper import FrozenLakeRewardShapingWrapper
        env = FrozenLakeRewardShapingWrapper(env)
    if one_hot:
        try:
            from DQN.onehot_wrapper import OneHotObservationWrapper
        except ImportError:
            from onehot_wrapper import OneHotObservationWrapper
        env = OneHotObservationWrapper(env)
    return env


DEFAULT_DQN_KWARGS = {
    "learning_rate": 0.0005,
    "buffer_size": 10000,
    "batch_size": 64,
    "gamma": 0.99,
    "target_update_interval": 100,
    "exploration_initial_eps": 0.5,
    "exploration_final_eps": 0.01,
    "exploration_fraction": 0.1,
    "verbose": 1,
}


def train_agent(
    total_timesteps,
    map_desc=None,
    max_episode_steps=None,
    is_slippery=False,
    reward_shaping: bool = False,
    one_hot: bool = False,
    **dqn_kwargs,
):
    """Return (model, logger). Pass DQN hyperparameters via **dqn_kwargs."""
    env = Monitor(
        make_env(
            map_desc=map_desc,
            is_slippery=is_slippery,
            max_episode_steps=max_episode_steps,
            reward_shaping=reward_shaping,
            one_hot=one_hot,
        )
    )
    horizon = int(max_episode_steps) if max_episode_steps is not None else 100
    logger = _TrainLogger(max_episode_steps=horizon)

    model = DQN("MlpPolicy", env, **{**DEFAULT_DQN_KWARGS, **dqn_kwargs})
    model.learn(total_timesteps=total_timesteps, callback=logger)
    env.close()
    return model, logger


def test_agent(
    model,
    num_episodes,
    map_desc=None,
    max_episode_steps=None,
    is_slippery=False,
    render_mode=None,
    reward_shaping: bool = False,
    one_hot: bool = False,
):
    env = make_env(
        render_mode=render_mode,
        map_desc=map_desc,
        is_slippery=is_slippery,
        max_episode_steps=max_episode_steps,
        reward_shaping=reward_shaping,
        one_hot=one_hot,
    )
    rewards = []
    successes = 0
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        last_info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        if _TrainLogger._episode_success_from_info(last_info, total_reward):
            successes += 1
        print(f"Episode {episode + 1}: reward={total_reward}")
    avg_reward = float(np.mean(rewards))
    success_rate = successes / num_episodes
    print("\n=== Evaluation Summary ===")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Success rate: {success_rate:.4f}")
    env.close()
    return avg_reward, success_rate


if __name__ == "__main__":
    try:
        from DQN.DQN_plots import save_training_visualization
    except ImportError:
        from DQN_plots import save_training_visualization

    # ----- Map config: only change MAP_SIZE (and IS_SLIPPERY if needed) -----
    # Plots + CSVs auto-save to: dqn_plots/<NxN>/  e.g. MAP_SIZE=4 -> dqn_plots/4x4/
    # Use MAP_SIZE=None for Gym's built-in default 4x4 (no maps.json); folder -> dqn_plots/gym_default/
    MAP_SIZE = 8
    IS_SLIPPERY = True
    USE_ONE_HOT = True
    USE_REWARD_SHAPING = False

    plot_subdir = plot_subdir_for_map_size(MAP_SIZE)
    if MAP_SIZE is None:
        map_desc, horizon = None, None
        plot_map_label = 4
    else:
        map_desc = load_map(MAP_SIZE)
        horizon = max_episode_steps_for_map_size(MAP_SIZE)
        plot_map_label = MAP_SIZE

    TS = 200_000
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PLOT_DIR = Path(__file__).resolve().parent / "dqn_plots" / plot_subdir
    if USE_ONE_HOT:
        PLOT_DIR = PLOT_DIR / "onehot"
    if USE_REWARD_SHAPING:
        PLOT_DIR = PLOT_DIR / "shaped"
    print(f"MAP_SIZE={MAP_SIZE!r} -> output directory:\n  {PLOT_DIR.resolve()}\n")

    model, logger = train_agent(
        TS,
        map_desc=map_desc,
        max_episode_steps=horizon,
        is_slippery=IS_SLIPPERY,
        reward_shaping=USE_REWARD_SHAPING,
        one_hot=USE_ONE_HOT,
    )
    save_training_visualization(
        logger,
        model,
        TS,
        str(PLOT_DIR),
        map_size=plot_map_label,
        is_slippery=IS_SLIPPERY,
    )
    test_agent(
        model,
        num_episodes=1000,
        map_desc=map_desc,
        max_episode_steps=horizon,
        is_slippery=IS_SLIPPERY,
        reward_shaping=USE_REWARD_SHAPING,
        one_hot=USE_ONE_HOT,
    )
