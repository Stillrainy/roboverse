import os
import glob
import numpy as np
import gymnasium as gym
from datetime import datetime

import roboverse
from roboverse.agents.offline_rls import OfflineAgentTrainer

data_dir = 'data/data_Widow250DrawerRandomizedPickPlace-v0_100_noise_0.1_2025-11-23T17-17-43'
save_dir = os.path.join(os.getcwd(), 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(save_dir, exist_ok=True)

trainer = OfflineAgentTrainer(
    data_dir=data_dir,
    obs_keys=['image', 'depth'],
    log_dir=save_dir,
    n_envs=10
)

trainer.train(
    total_timesteps=100_000,
    progress_bar=True,
)

