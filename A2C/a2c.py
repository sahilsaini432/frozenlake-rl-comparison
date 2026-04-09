import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Maps ───────────────────────────────────────────────────────────────────────

four_x_four_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]

eight_x_eight_map = [
    "SFFFFFFH",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFFFFFF",
    "FFFFFFFF",
    "FFHHHFFF",
    "FHFFFFFF",
    "FHFFFFFG",
]

sixteen_x_sixteen_map = [
    "SFFHFFFHFFFHFFFF",
    "FHFFFHFFHFFHFFHF",
    "FFHFFFFFHFFHFFFF",
    "HFFFHFFHFFFHFHFF",
    "FFFHFFFHFHFFHFFF",
    "FHFFHFFFHFFFHFHF",
    "FFFHFFFFHFHFFHFF",
    "HFFFHFHFFFHFFFHF",
    "FFHFFFHFHFFFHFFF",
    "FHFFHFFFHFHFFHFF",
    "FFFHFFFHFFHFHFFF",
    "HFHFFFHFHFFFHFFH",
    "FFFHFHFFFHFFHFFF",
    "FHFFFHFFHFHFFFHF",
    "FFHFHFFFHFFFHFFF",
    "HFFFHFHFFFHFFHFG",
]

# ── One-hot wrapper ────────────────────────────────────────────────────────────

class OneHotObservationWrapper(ObservationWrapper):
    """Converts FrozenLake's discrete integer state into a one-hot float vector."""
    def __init__(self, env):
        super().__init__(env)
        n = env.observation_space.n
        self.n_states = n
        self.observation_space = Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


# ── Env factory ───────────────────────────────────────────────────────────────

def make_env(desc, is_slippery=True):
    """Returns a callable that builds one wrapped + monitored env instance."""
    def _init():
        e = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery)
        e = Monitor(e)
        e = OneHotObservationWrapper(e)
        return e
    return _init


# ── Monitor helpers ───────────────────────────────────────────────────────────

def get_monitor(vec_env, idx: int = 0) -> Monitor:
    """Walk the wrapper stack inside a DummyVecEnv to find the Monitor."""
    layer = vec_env.envs[idx]
    while not isinstance(layer, Monitor):
        if not hasattr(layer, "env"):
            raise ValueError("No Monitor found in wrapper stack.")
        layer = layer.env
    return layer


def smooth(values, window: int = 100):
    """Rolling mean; returns values unchanged if shorter than window."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_monitor(vec_env, window: int = 200):
    """Four diagnostic plots sourced from the Monitor inside vec_env."""
    mon = get_monitor(vec_env)
    rewards = np.array(mon.get_episode_rewards())
    lengths = np.array(mon.get_episode_lengths())

    if len(rewards) == 0:
        print("No episodes recorded yet.")
        return

    ep = np.arange(len(rewards))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Monitor diagnostics", fontsize=14)

    # 1. Success rate
    ax = axes[0, 0]
    ax.plot(ep, rewards, alpha=0.15, color="steelblue", linewidth=0.5, label="raw")
    s = smooth(rewards, window)
    if len(s) > 0:
        offset = (len(rewards) - len(s)) // 2
        ax.plot(np.arange(len(s)) + offset, s,
                color="steelblue", linewidth=1.5, label=f"{window}-ep rolling mean")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (0 or 1)")
    ax.set_title("Success rate over episodes")
    ax.legend(fontsize=8)

    # 2. Episode length
    ax = axes[0, 1]
    ax.plot(ep, lengths, alpha=0.15, color="darkorange", linewidth=0.5, label="raw")
    s = smooth(lengths, window)
    if len(s) > 0:
        offset = (len(lengths) - len(s)) // 2
        ax.plot(np.arange(len(s)) + offset, s,
                color="darkorange", linewidth=1.5, label=f"{window}-ep rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode length")
    ax.legend(fontsize=8)

    # 3. Cumulative successes
    ax = axes[1, 0]
    ax.plot(ep, np.cumsum(rewards), color="seagreen", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total successes")
    ax.set_title("Cumulative successes")

    # 4. Length vs outcome scatter
    ax = axes[1, 1]
    ax.scatter(lengths[rewards == 0], ep[rewards == 0],
               alpha=0.3, s=4, color="tomato",   label="failure")
    ax.scatter(lengths[rewards == 1], ep[rewards == 1],
               alpha=0.5, s=6, color="seagreen", label="success")
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel("Episode")
    ax.set_title("Length vs outcome")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_training_curve(timesteps, success_rates):
    """Plot the held-out eval success rate vs training timesteps."""
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, success_rates, marker="o", markersize=4, color="steelblue")
    plt.xlabel("Training timesteps")
    plt.ylabel("Success rate (100 eval episodes)")
    plt.title("A2C on FrozenLake — eval curve")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Setup ─────────────────────────────────────────────────────────────────────

MAP = eight_x_eight_map

# Separate envs: one for training (records all episodes), one for clean eval
train_env = DummyVecEnv([make_env(MAP)])
eval_env  = make_env(MAP)()          # plain unwrapped instance for eval loop

model = A2C(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=0.0007,
    n_steps=128,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
)

# ── Training loop ─────────────────────────────────────────────────────────────

CHUNK      = 25_000
N_CHUNKS   = 60         # 300k total timesteps
N_EVAL     = 100         # eval episodes per checkpoint

timesteps     = []
success_rates = []
total_trained = 0

for i in range(N_CHUNKS):
    model.learn(total_timesteps=CHUNK, reset_num_timesteps=False)
    total_trained += CHUNK

    successes = 0
    for _ in range(N_EVAL):
        obs, _ = eval_env.reset()
        done = truncated = False
        reward = 0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = eval_env.step(int(action))
        if reward == 1.0:
            successes += 1

    rate = successes / N_EVAL
    timesteps.append(total_trained)
    success_rates.append(rate)
    print(f"Chunk {i+1:>2}/{N_CHUNKS}  |  {total_trained:>7,} steps  |  eval success: {rate:.0%}")

# ── Results ───────────────────────────────────────────────────────────────────

plot_training_curve(timesteps, success_rates)
plot_monitor(train_env, window=200)