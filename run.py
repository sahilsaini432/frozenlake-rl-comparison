# Update this file to incorporate any new agent or implementation you create

import argparse
from time import sleep
import gymnasium as gym
from stable_baselines3 import A2C
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from A2C.base_A2C import ModA2C
from PPO.test_PPO import ModPPO



def selected_agent(args):
    if args.agent == "PPO":
        return ModPPO("MlpPolicy", env)
    elif args.agent == "A2C":
        return ModA2C("MlpPolicy", env)


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--episodes", type=int, default=100, help="Number of training episodes")
parser.add_argument("-a", "--agent", type=int, default=100, help="agent selected")
args = parser.parse_args()

# Set up the environment
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=8), is_slippery=True, render_mode="human")

# Initialize agent
agent = selected_agent(args)

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
        # wait for 3 sec
        sleep(3)
    obs, info = env.reset()

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
