import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def moving_average(x, w):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def plot_config_curves(base_dir, config_name, seeds, window=100, out_dir='.'):
    fig, ax = plt.subplots(figsize=(10, 6))
    found_any = False

    for seed in seeds:
        npz_path = os.path.join(base_dir, config_name, f'seed{seed}', f'dqn_data_{config_name}_seed{seed}.npz')
        if not os.path.exists(npz_path):
            print(f'[WARN] Missing file: {npz_path}')
            continue

        data = np.load(npz_path)
        timesteps = data['timesteps']
        rewards = data['rewards']

        if len(rewards) == 0:
            print(f'[WARN] No reward data in: {npz_path}')
            continue

        found_any = True
        if len(rewards) >= window:
            smoothed = moving_average(rewards, window)
            ax.plot(timesteps[window - 1:], smoothed, linewidth=2, label=f'Seed {seed}')
        else:
            ax.plot(timesteps, rewards, linewidth=1.5, label=f'Seed {seed}')

    if not found_any:
        print(f'[WARN] No valid data found for {config_name}')
        plt.close(fig)
        return

    ax.set_title(f'DQN Learning Curves - {config_name}')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel(f'Episode Reward Moving Average ({window})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{config_name}_three_seed_learning_curves.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Plot one 3-seed learning-curve graph per DQN config.')
    parser.add_argument('--base_dir', type=str, default='.', help='Directory containing config1/config2/... folders')
    parser.add_argument('--configs', nargs='+', default=['config1', 'config2', 'config3', 'config4'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--out_dir', type=str, default='.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for config_name in args.configs:
        plot_config_curves(args.base_dir, config_name, args.seeds, args.window, args.out_dir)


if __name__ == '__main__':
    main()
