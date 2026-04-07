import json
import os

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .base_PPO import ModPPO
from .callbacks import TrainingLoggerCallback
from .metrics import compute_metrics, evaluate_agent
from .onehot_wrapper import OneHotWrapper
from .plots import (
    plot_approx_kl,
    plot_entropy_loss,
    plot_episode_length,
    plot_training_curve,
    save_aggregate_summary_table,
    save_summary_table,
)
from .reward_shaping_wrapper import RewardShapingWrapper
from .naming import pretty_run_name


def make_frozenlake_env(
    is_slippery=True,
    custom_map=None,
    map_size=8,
    step_penalty=0.0,
    manhattan_scale=0.0,
):
    def _init():
        if custom_map is not None:
            env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=is_slippery)
        else:
            env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)

        env = OneHotWrapper(env)

        if step_penalty > 0.0 or manhattan_scale > 0.0:
            env = RewardShapingWrapper(
                env,
                map_size=map_size,
                step_penalty=step_penalty,
                manhattan_scale=manhattan_scale,
            )

        env = Monitor(env)
        return env

    return _init


def load_map(map_size):
    maps_path = os.path.join(os.path.dirname(__file__), "..", "maps.json")
    maps_path = os.path.abspath(maps_path)  
    
    with open(maps_path, "r") as file:
        all_maps = json.load(file)

    map_key = str(map_size)
    if map_key not in all_maps:
        raise ValueError(
            f"Map size {map_size} not found in maps.json. Available: {list(all_maps.keys())}"
        )

    return all_maps[map_key]


def build_config(args):
    is_slippery = not args.deterministic
    custom_map = load_map(args.map_size)

    return {
        "is_slippery": is_slippery,
        "hidden_size": args.hidden_size,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "timesteps": args.timesteps,
        "n_eval": args.n_eval,
        "ent_coef": args.ent_coef,
        "gae_lambda": args.gae_lambda,
        "vf_coef": args.vf_coef,
        "clip_range": args.clip_range,
        "step_penalty": args.step_penalty,
        "manhattan_scale": args.manhattan_scale,
        "custom_map": custom_map,
        "map_size": args.map_size,
    }


def get_output_root(args):
    if args.output_dir is not None:
        return args.output_dir

    base_output_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        f"{args.map_size}x{args.map_size}",
    )

    if args.run_name:
        return os.path.join(base_output_dir, args.run_name)

    return base_output_dir


def print_run_config(config, args, run_root):
    mode = "stochastic" if config["is_slippery"] else "deterministic"

    print("=" * 60)
    print(f"PPO on FrozenLake-v1 ({mode})")
    print("=" * 60)
    print(f"  Map:             {args.map_size}x{args.map_size} (custom)")
    print(f"  Slippery:        {config['is_slippery']}")
    print(f"  Hidden size:     {args.hidden_size}")
    print(f"  n_steps:         {args.n_steps}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Clip range:      {args.clip_range}")
    print(f"  GAE lambda:      {args.gae_lambda}")
    print(f"  vf_coef:         {args.vf_coef}")
    print(f"  Timesteps:       {args.timesteps}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Eval episodes:   {args.n_eval}")
    print(f"  ent_coef:        {args.ent_coef}")
    print(f"  Step penalty:    {args.step_penalty}")
    print(f"  Manhattan scale: {args.manhattan_scale}")
    print(f"  Run name:        {args.run_name or '(none)'}")
    print(f"  Output root:     {run_root}")
    print("=" * 60)


def run_single_seed(seed, config, output_dir, label_name):
    os.makedirs(output_dir, exist_ok=True)

    is_slippery = config["is_slippery"]
    hidden_size = config["hidden_size"]
    timesteps = config["timesteps"]
    n_eval = config["n_eval"]

    print(f"\nSeed {seed}")
    pretty_name = pretty_run_name(label_name)

    train_env_fn = make_frozenlake_env(
        is_slippery=is_slippery,
        custom_map=config.get("custom_map"),
        map_size=config.get("map_size", 8),
        step_penalty=config.get("step_penalty", 0.0),
        manhattan_scale=config.get("manhattan_scale", 0.0),
    )

    eval_env_fn = make_frozenlake_env(
        is_slippery=is_slippery,
        custom_map=config.get("custom_map"),
        map_size=config.get("map_size", 8),
        step_penalty=0.0,
        manhattan_scale=0.0,
    )

    model = ModPPO(
        policy="MlpPolicy",
        env=DummyVecEnv([train_env_fn]),
        learning_rate=config["lr"],
        n_steps=config["n_steps"],
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        clip_range_vf=None,
        ent_coef=config.get("ent_coef", 0.0),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=0.5,
        seed=seed,
        verbose=0,
        tensorboard_log=os.path.join(output_dir, "ppo_tensorboard"),
        policy_kwargs=dict(
            net_arch=dict(
                pi=[hidden_size, hidden_size],
                vf=[hidden_size, hidden_size],
            ),
        ),
    )

    callback = TrainingLoggerCallback()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    model.save(os.path.join(output_dir, f"ppo_seed{seed}"))

    timesteps_arr = np.array(callback.timesteps_at_episode)
    rewards_arr = np.array(callback.episode_rewards)
    lengths_arr = np.array(callback.episode_lengths)

    eval_rewards = evaluate_agent(model, eval_env_fn, n_episodes=n_eval)
    metrics = compute_metrics(eval_rewards)
    train_reward_l100 = float("nan")
    if len(rewards_arr) >= 100:
        train_reward_l100 = np.mean(rewards_arr[-100:])
    
    episode_length_l100 = float("nan")
    if len(lengths_arr) >= 100:
        episode_length_l100 = np.mean(lengths_arr[-100:])
    
    final_kl = callback.approx_kls[-1] if callback.approx_kls else float("nan")

    plot_training_curve(
        timesteps_arr,
        rewards_arr,
        window=100,
        title=f"{pretty_name} Learning Curve (seed={seed})",
        filename=os.path.join(output_dir, f"ppo_training_curve_seed{seed}.png"),
    )

    if callback.entropy_losses:
        plot_entropy_loss(
            callback.training_timesteps,
            callback.entropy_losses,
            title=f"{pretty_name} Entropy Loss (seed={seed})",
            filename=os.path.join(output_dir, f"ppo_entropy_loss_seed{seed}.png"),
        )

    if callback.approx_kls:
        plot_approx_kl(
            callback.training_timesteps,
            callback.approx_kls,
            title=f"{pretty_name} Stability Signal (seed={seed})",
            filename=os.path.join(output_dir, f"ppo_approx_kl_seed{seed}.png"),
        )

    if len(callback.episode_lengths) >= 100:
        plot_episode_length(
            timesteps_arr,
            lengths_arr,
            window=100,
           title=f"{pretty_name} Episode Length Curve (seed={seed})",
            filename=os.path.join(output_dir, f"ppo_episode_length_seed{seed}.png"),
        )

    save_summary_table(
        config,
        metrics,
        seed,
        pretty_name,
        train_reward_l100,
        episode_length_l100,
        final_kl,
        filename=os.path.join(output_dir, f"ppo_summary_seed{seed}.png"),
    )

    np.savez(
        os.path.join(output_dir, f"ppo_data_seed{seed}.npz"),
        timesteps=timesteps_arr,
        rewards=rewards_arr,
        lengths=lengths_arr,
    )

    print(
        f"  Success Rate: {metrics['success_rate']:.1%} | "
        f"Reward Std: {metrics['eval_std']:.3f} | "
        f"Train Reward (last 100): {train_reward_l100:.3f} | "
        f"Episode Length (last 100): {episode_length_l100:.1f} | "
        f"Final KL: {final_kl:.5f}"
)

    return metrics


def run_experiment(args):
    config = build_config(args)
    run_root = get_output_root(args)
    label_name = args.run_name if args.run_name else "baseline"
    pretty_name = pretty_run_name(label_name)

    print_run_config(config, args, run_root)

    all_metrics = []

    for seed in args.seeds:
        seed_dir = os.path.join(run_root, f"seed{seed}")
        metrics = run_single_seed(seed, config, seed_dir, label_name)
        all_metrics.append(metrics)

    if len(all_metrics) > 1:
        success_rates = [metric["success_rate"] for metric in all_metrics]

        print(f"\n{'=' * 60}")
        print(f"AGGREGATE ({len(args.seeds)} seeds)")
        print(f"{'=' * 60}")
        print(f"  Agg Success Rate: {np.mean(success_rates):.1%} +/- {np.std(success_rates):.1%}")
        print(f"{'=' * 60}")

        save_aggregate_summary_table(
            config,
            all_metrics,
            args.seeds,
            pretty_name,
            filename=os.path.join(run_root, "ppo_aggregate_summary.png"),
        )