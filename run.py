# Update this file to incorporate any new agent or implementation you create

import argparse
from time import sleep
import gymnasium as gym
from stable_baselines3 import A2C
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from A2C.base_A2C import ModA2C
from MCTS.mcts import MCTS
from MCTS.mcts_base import MCTSBase


def selected_agent(args):
    if args.mcts_base:
        # Using the specs from the project proposal for MCTS
        return MCTSBase(env=env, num_simulations=1000, exploration_constant=1.4, max_rollout_depth=100)
    return ModA2C("MlpPolicy", env)


def evaluate_mcts(env, agent, episodes):
    """Run MCTS for a fixed number of episodes and report success rate and avg reward."""
    total_rewards = []
    successes = 0

    for episode in tqdm(range(episodes), desc="Evaluating MCTS"):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.search(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        if episode_reward > 0:
            successes += 1

    print(f"\n=== MCTS Evaluation over {episodes} episodes ===")
    print(f"Success rate:   {successes / episodes * 100:.1f}%")
    print(f"Avg reward:     {sum(total_rewards) / episodes:.4f}")
    print(f"Min/Max reward: {min(total_rewards):.4f} / {max(total_rewards):.4f}")
    return total_rewards


def train_agent(env, agent, episodes):
    if isinstance(agent, (MCTS, MCTSBase)):
        evaluate_mcts(env, agent, episodes)
    else:
        # Train the agent
        num_episodes = args.episodes
        obs, info = env.reset()

        for episode in tqdm(range(num_episodes), desc="Training"):
            done = False
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.learn(obs, action, reward, next_obs, terminated)
                obs = next_obs
                done = terminated or truncated
                sleep(3)
            obs, info = env.reset()


def test_agent(env, agent):
    # Test the trained agent
    obs, info = env.reset()
    for step in tqdm(range(1000)):
        action = agent.select_action(obs)
        print(f"\n--- Step {step + 1} ---")
        print(f"Current observation: {obs}")
        print(f"Action taken: {action}")

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Next observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            obs, info = env.reset()
            print(f"Environment reset. New observation: {obs}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--mcts_base", action="store_true", help="Use MCTS as the base agent")
    args = parser.parse_args()

    # Set up the environment
    env = gym.make("FrozenLake-v1", desc=generate_random_map(size=8), is_slippery=True, render_mode="human")

    # Initialize agent
    agent = selected_agent(args)

    # Train the agent
    train_agent(env, agent, args.episodes)

    # Test the agent
    test_agent(env, agent)
