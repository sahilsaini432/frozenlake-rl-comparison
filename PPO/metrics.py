import numpy as np


def evaluate_agent(model, env_fn, n_episodes=1000):
    eval_env = env_fn()
    rewards = []

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = model.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)

    eval_env.close()
    return rewards


def compute_metrics(eval_rewards):
    return {
        "eval_mean": np.mean(eval_rewards),
        "eval_std": np.std(eval_rewards),
        "success_rate": np.mean([1 if reward > 0 else 0 for reward in eval_rewards]),
    }