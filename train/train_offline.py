import os
import glob
import numpy as np
import gymnasium as gym
import argparse
from datetime import datetime

import roboverse
from roboverse.agents import OfflineAgentTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train offline agent')
    parser.add_argument('-d', '--data-dir', type=str, required=True,
                        help='Path to input data directory')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory to write logs. If omitted, a timestamped folder under `logs/` in CWD will be used')
    parser.add_argument('--obs-keys', nargs='+', default=['image', 'depth'],
                        help='Observation keys to use (space-separated). Default: %(default)s')
    parser.add_argument('-s', '--save-freq', type=int, default=1000,
                        help='Model save frequency (in steps)')
    parser.add_argument('-t', '--total-timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bar (enabled by default)')
    parser.add_argument('-i', '--log-interval', type=int, default=None,
                        help='Logging interval (None for default behavior)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # compute default log dir if not provided
    if args.log_dir is None:
        save_dir = os.path.join(
            os.getcwd(),
            'logs',
            f'offline_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    else:
        save_dir = args.log_dir
    os.makedirs(save_dir, exist_ok=True)

    trainer = OfflineAgentTrainer(
        data_dir=args.data_dir,
        obs_keys=args.obs_keys,
        log_dir=save_dir,
        save_freq=args.save_freq,
    )

    trainer.train(
        total_timesteps=args.total_timesteps,
        progress_bar=(not args.no_progress),
        log_interval=args.log_interval,
    )
