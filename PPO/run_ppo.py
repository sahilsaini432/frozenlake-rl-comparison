import argparse

from .experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="PPO on FrozenLake-v1")

    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--step_penalty", type=float, default=0.0)
    parser.add_argument("--manhattan_scale", type=float, default=0.0)
    parser.add_argument("--map_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()