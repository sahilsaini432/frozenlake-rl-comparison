import gymnasium as gym
from stable_baselines3 import PPO
import torch
import numpy as np

# standalone PPO agent test on FrozenLake environment, just want to make sure everything
# works then can integrate it 


# Create FrozenLake environment
def make_env(render_mode=None):
    return gym.make("FrozenLake-v1", is_slippery=True, render_mode=render_mode)

# Create PPO agent and train it on the environment
def train_agent(total_timesteps = 50000):
    env = make_env()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model

def test_agent(model, num_episodes=10, render_mode=None):
    env = make_env(render_mode=render_mode)

    rewards = []
    successes = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)
        if total_reward > 0:
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
    model = train_agent(total_timesteps=10000)
    test_agent(model, num_episodes=20)



    