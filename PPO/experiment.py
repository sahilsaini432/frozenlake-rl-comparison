"""
Experiment script responsible for a lot of the heavy lifting around running PPO on FrozenLake, including:
Env setup, training loop, eval, metrics computations, calling plotting and table generation, and saving results
"""

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
from .plots import (plot_approx_kl, plot_entropy_loss, plot_episode_length, plot_training_curve, save_aggregate_summary_table, save_summary_table)
from .reward_shaping_wrapper import RewardShapingWrapper
from .naming import pretty_run_name


def make_env(is_slippery = True, custom_map = None, map_size = 8, step_penalty = 0.0, manhattan_scale = 0.0):
    def _init():
        if custom_map is not None:
            env = gym.make("FrozenLake-v1", desc = custom_map, is_slippery = is_slippery)
        else:
            env = gym.make("FrozenLake-v1", map_name = "4x4", is_slippery = is_slippery)
        env = OneHotWrapper(env)
        if step_penalty > 0.0 or manhattan_scale > 0.0:
            env = RewardShapingWrapper(env, map_size = map_size, step_penalty = step_penalty, manhattan_scale = manhattan_scale)
        env = Monitor(env)
        return env
    return _init

# Load the custom map from maps.json instead of generating a random one, all algos use same maps for comparison
def load_map(map_size):
    maps_path =  os.path.join(os.path.dirname(__file__), "..", "maps.json")
    maps_path = os.path.abspath(maps_path)
    with open(maps_path, "r") as f:
        all_maps = json.load(f)
    if str(map_size) not in all_maps:
        raise ValueError(
            f"Map size {map_size} not found in maps.json, Available: {list(all_maps.keys())}"
        )
    return all_maps[str(map_size)]

def build_cfg(args):
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
        "map_size": args.map_size
    }

def get_out_root(args):
    if args.output_dir is not None:
        return args.output_dir
    base_dir = os.path.join(os.path.dirname(__file__), "results", f"{args.map_size}x{args.map_size}")
    if args.run_name:
        return os.path.join(base_dir, args.run_name)
    return base_dir


def print_cfg(config, args, run_root):
    mode = "stochastic" if config["is_slippery"] else "deterministic"
    print()
    print(f"PPO on FrozenLake-v1 ({mode})")
    print()
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
    print()


def run_seed(seed, config, output_dir, label):
    os.makedirs(output_dir, exist_ok = True)
    is_slippery = config["is_slippery"]
    hidden_size = config["hidden_size"]
    timesteps = config["timesteps"]
    n_eval = config["n_eval"]
    print(f"\nSeed {seed}")
    pname = pretty_run_name(label)
 
    # Train env uses shaping if configured, and  eval env does not get for comparison 
    train_fn = make_env(
        is_slippery = is_slippery,
        custom_map = config.get("custom_map"),
        map_size = config.get("map_size", 8),
        step_penalty = config.get("step_penalty", 0.0),
        manhattan_scale = config.get("manhattan_scale", 0.0),
    )
    eval_fn  = make_env(
        is_slippery = is_slippery,
        custom_map = config.get("custom_map"),
        map_size = config.get("map_size", 8),
        step_penalty = 0.0,
        manhattan_scale = 0.0
    )

    model = ModPPO(
        policy = "MlpPolicy",
        env = DummyVecEnv([train_fn]),
        learning_rate = config["lr"],
        n_steps = config["n_steps"],
        batch_size = 64,
        n_epochs = 10,
        gamma = 0.99,
        gae_lambda = config.get("gae_lambda", 0.95),
        clip_range = config.get("clip_range", 0.2),
        clip_range_vf = None,
        ent_coef = config.get("ent_coef", 0.0),
        vf_coef = config.get("vf_coef", 0.5),
        max_grad_norm = 0.5,
        seed = seed,
        verbose = 0,
        tensorboard_log = os.path.join(output_dir, "ppo_tensorboard"),
        policy_kwargs = dict( net_arch = dict(pi = [hidden_size, hidden_size], vf = [hidden_size, hidden_size]))
    )
 
    cb = TrainingLoggerCallback()
    model.learn(total_timesteps = timesteps, callback = cb, progress_bar = True)
    model.save(os.path.join(output_dir, f"ppo_seed{seed}"))
 
    ts = np.array(cb.ts_e)
    rews = np.array(cb.ep_rewards)
    lens = np.array(cb.ep_lens)
 
    eval_rew = evaluate_agent(model, eval_fn, n_episodes = n_eval)
    metrics  = compute_metrics(eval_rew)
 
    train_r100 = float("nan")
    if len(rews) >= 100:
        train_r100 = np.mean(rews[-100:])
    ep_len100 = float("nan")
    if len(lens) >= 100:
        ep_len100 = np.mean(lens[-100:])
    kl_final = cb.approx_kls[-1] if cb.approx_kls else float("nan")
 
    plot_training_curve(ts, rews, window = 100, title = f"{pname} Learning Curve (seed={seed})",
        filename = os.path.join(output_dir, f"ppo_training_curve_seed{seed}.png")
    )
    if cb.en_loss:
        plot_entropy_loss(cb.train_ts, cb.en_loss, title = f"{pname} Entropy Loss (seed={seed})",
            filename = os.path.join(output_dir, f"ppo_entropy_loss_seed{seed}.png")
        )
    if cb.approx_kls:
        plot_approx_kl(cb.train_ts, cb.approx_kls,title = f"{pname} Stability Signal (seed={seed})",
            filename = os.path.join(output_dir, f"ppo_approx_kl_seed{seed}.png"))
        
    if len(cb.ep_lens) >= 100:
        plot_episode_length(ts, lens, window = 100, title = f"{pname} Episode Length Curve (seed={seed})",
            filename = os.path.join(output_dir, f"ppo_episode_length_seed{seed}.png")
        )

    save_summary_table(config, metrics, seed, pname, train_r100, ep_len100, kl_final,
        filename = os.path.join(output_dir, f"ppo_summary_seed{seed}.png"))
    np.savez(
        os.path.join(output_dir, f"ppo_data_seed{seed}.npz"),
        timesteps = ts, rewards = rews, lengths = lens)
    
    print(
        f"  Success Rate: {metrics['success_rate']:.1%} | "
        f"Reward Std: {metrics['eval_std']:.3f} | "
        f"Train Reward (last 100): {train_r100:.3f} | "
        f"Episode Length (last 100): {ep_len100:.1f} | "
        f"Final KL: {kl_final:.5f}"
    )
    return metrics
 
 
def run_experiment(args):
    config= build_cfg(args)
    run_root = get_out_root(args)
    if args.run_name:
        label = args.run_name
    else:
        label = "baseline"
    pname = pretty_run_name(label)
    print_cfg(config, args, run_root)
 
    all_metrics = []
    for seed in args.seeds:
        seed_dir = os.path.join(run_root, f"seed{seed}")
        m = run_seed(seed, config, seed_dir, label)
        all_metrics.append(m)
 
    if len(all_metrics) > 1:
        s_rates = []
        for m in all_metrics:
            s_rates.append(m["success_rate"])
        print()
        print(f"AGGREGATE ({len(args.seeds)} seeds)")
        print(f"Agg Success Rate: {np.mean(s_rates):.1%} +/- {np.std(s_rates):.1%}")
        print()
        save_aggregate_summary_table(config, all_metrics, args.seeds, pname, filename = os.path.join(run_root, "ppo_aggregate_summary.png"))