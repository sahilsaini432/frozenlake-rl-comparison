import argparse
import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, spaces
from stable_baselines3 import PPO


class OneHotObservationWrapper(ObservationWrapper):
    """
    Convert discrete state index into one-hot vector.
    Example: state 5 in 16 states -> [0,0,0,0,0,1,0,...]
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Discrete), \
            "OneHotObservationWrapper only supports Discrete observation spaces."

        self.n_states = env.observation_space.n
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_states,),
            dtype=np.float32
        )

    def observation(self, obs):
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[int(obs)] = 1.0
        return one_hot


def build_modified_frozenlake(
    map_name="8x8",
    is_slippery=True,
    slip_prob=0.2,
    render_mode=None
):
    """
    Create FrozenLake and modify transition probabilities so that:
      - intended action probability = 1 - slip_prob
      - side slip probabilities share slip_prob equally
    For FrozenLake with 4 actions, the slipping is to the left/right neighboring
    actions of the intended action.
    """
    env = gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode=render_mode
    )

    # Unwrap to access underlying FrozenLake env internals
    unwrapped = env.unwrapped

    # If slippery is disabled, nothing to modify
    if not is_slippery:
        return env

    nA = unwrapped.action_space.n
    nS = unwrapped.observation_space.n

    # Proposal says slip probability ≈ 0.2
    # So intended action = 0.8, slips = 0.1 and 0.1
    intended_prob = 1.0 - slip_prob
    side_prob = slip_prob / 2.0

    # Rebuild transition matrix P
    new_P = {s: {a: [] for a in range(nA)} for s in range(nS)}

    # Original FrozenLake helper data
    desc = unwrapped.desc
    nrow, ncol = desc.shape

    def to_s(row, col):
        return row * ncol + col

    def inc(row, col, action):
        if action == 0:   # LEFT
            col = max(col - 1, 0)
        elif action == 1: # DOWN
            row = min(row + 1, nrow - 1)
        elif action == 2: # RIGHT
            col = min(col + 1, ncol - 1)
        elif action == 3: # UP
            row = max(row - 1, 0)
        return row, col

    def update_probability_matrix(row, col, action):
        new_row, new_col = inc(row, col, action)
        new_state = to_s(new_row, new_col)
        new_letter = desc[new_row, new_col].decode("utf-8")
        terminated = new_letter in "HG"
        reward = float(new_letter == "G")  # sparse reward: only goal gives 1
        return new_state, reward, terminated

    for row in range(nrow):
        for col in range(ncol):
            s = to_s(row, col)
            letter = desc[row, col].decode("utf-8")

            for a in range(nA):
                li = new_P[s][a]

                # Terminal states stay terminal
                if letter in "HG":
                    li.append((1.0, s, 0.0, True))
                else:
                    left_action = (a - 1) % 4
                    intended_action = a
                    right_action = (a + 1) % 4

                    for prob, actual_action in [
                        (side_prob, left_action),
                        (intended_prob, intended_action),
                        (side_prob, right_action),
                    ]:
                        new_state, reward, terminated = update_probability_matrix(
                            row, col, actual_action
                        )
                        li.append((prob, new_state, reward, terminated))

    unwrapped.P = new_P
    return env


def make_env(map_name="8x8", slip_prob=0.2, render_mode=None):
    env = build_modified_frozenlake(
        map_name=map_name,
        is_slippery=True,
        slip_prob=slip_prob,
        render_mode=render_mode
    )
    env = OneHotObservationWrapper(env)
    return env


def create_ppo_model(env, hidden_size=16, learning_rate=3e-4, verbose=1):
    """
    Proposal: feedforward neural network with two hidden layers,
    tested hidden sizes of 2, 4, 8, 16 neurons per layer.
    """
    policy_kwargs = dict(
        net_arch=[hidden_size, hidden_size]
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        verbose=verbose
    )
    return model


def train_agent(total_timesteps=10000, hidden_size=16, slip_prob=0.2):
    env = make_env(map_name="8x8", slip_prob=slip_prob, render_mode=None)
    model = create_ppo_model(env, hidden_size=hidden_size, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_agent(model, num_episodes=20, slip_prob=0.2, render_mode=None):
    env = make_env(map_name="8x8", slip_prob=slip_prob, render_mode=render_mode)

    rewards = []
    successes = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        if total_reward > 0:
            successes += 1

        print(f"Episode {episode + 1}: reward={total_reward}, steps={steps}")

    avg_reward = float(np.mean(rewards))
    success_rate = successes / num_episodes

    print("\n=== Evaluation Summary ===")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Success rate: {success_rate:.4f}")

    env.close()
    return avg_reward, success_rate


def run_all_architectures(total_timesteps=10000, slip_prob=0.2):
    """
    Run the 4 proposed hidden sizes: 2, 4, 8, 16
    """
    results = {}

    for hidden_size in [2, 4, 8, 16]:
        print(f"\n{'=' * 60}")
        print(f"Training PPO with hidden size = {hidden_size}")
        print(f"{'=' * 60}")

        model = train_agent(
            total_timesteps=total_timesteps,
            hidden_size=hidden_size,
            slip_prob=slip_prob
        )

        avg_reward, success_rate = evaluate_agent(
            model,
            num_episodes=20,
            slip_prob=slip_prob,
            render_mode=None
        )

        results[hidden_size] = {
            "avg_reward": avg_reward,
            "success_rate": success_rate
        }

    print("\n=== Final Comparison ===")
    for hidden_size, result in results.items():
        print(
            f"Hidden size {hidden_size}: "
            f"avg_reward={result['avg_reward']:.4f}, "
            f"success_rate={result['success_rate']:.4f}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="Total training timesteps")
    parser.add_argument("--hidden", type=int, default=16, choices=[2, 4, 8, 16],
                        help="Hidden layer size for both hidden layers")
    parser.add_argument("--slip_prob", type=float, default=0.2,
                        help="Slip probability (proposal uses about 0.2)")
    parser.add_argument("--eval_episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render evaluation episodes")
    parser.add_argument("--all_architectures", action="store_true",
                        help="Run experiments for hidden sizes 2,4,8,16")

    args = parser.parse_args()

    if args.all_architectures:
        run_all_architectures(
            total_timesteps=args.timesteps,
            slip_prob=args.slip_prob
        )
    else:
        model = train_agent(
            total_timesteps=args.timesteps,
            hidden_size=args.hidden,
            slip_prob=args.slip_prob
        )

        evaluate_agent(
            model,
            num_episodes=args.eval_episodes,
            slip_prob=args.slip_prob,
            render_mode="human" if args.render else None
        )