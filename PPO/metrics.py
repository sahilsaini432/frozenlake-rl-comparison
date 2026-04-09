"""
Metrics for agent eval and computation 
"""

import numpy as np

def evaluate_agent(model, env_fn, n_episodes = 1000):
    env = env_fn()
    rews = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action = model.select_action(obs, deterministic = True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            done = terminated or truncated
        rews.append(ep_rew)
    env.close()
    return rews

def compute_metrics(eval_rew):
    return {
        "eval_mean": np.mean(eval_rew),
        "eval_std": np.std(eval_rew),
        "success_rate": np.mean([1 if r > 0 else 0 for r in eval_rew])
    }