import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple
from stable_baselines3 import (
    DDPG,
    TD3,
    SAC
)
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)

from .utils import (
    load_dataset_from_dir,
    build_space,
    ImageDepthEnvWrapper
)
from .custom_cnn import CustomCNN


class OfflineAgentTrainer:
    def __init__(
        self,
        data_dir: str,
        obs_keys: List[str],
        algo: str = "SAC",
        device: str = "cuda",
        log_dir: str = "./logs",
        save_freq: int = 10_000,
    ):
        self.obs_keys = obs_keys
        self.log_dir = log_dir
        self.save_freq = save_freq

        # Load dataset
        obs, act, rew, next_obs, dones = load_dataset_from_dir(
            data_dir, obs_keys)

        # Build spaces
        self.eval_env = gym.make(data_dir.split('_')[1])
        obs_space, action_space = build_space(self.eval_env, obs_keys)

        # Create offline replay env
        self.env = OfflineReplayEnv(
            obs, act, rew, next_obs, dones,
            obs_space, action_space
        )

        # Convert to Box observation if using image/depth only
        if obs_keys and any(k in ["image", "depth"] for k in obs_keys):
            self.env = ImageDepthEnvWrapper(self.env)
            self.eval_env = ImageDepthEnvWrapper(self.eval_env)

        # Auto-select policy type
        obs_space = self.env.observation_space
        if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
            # (C,H,W) tensor → image
            self.policy_type = "CnnPolicy"
            normalize_images = False
        else:
            self.policy_type = "MultiInputPolicy"
            normalize_images = True

        # SB3 Policy kwargs
        policy_kwargs = dict(
            normalize_images=normalize_images,
            # features_extractor_class=CustomCNN,
            # features_extractor_kwargs=dict(features_dim=256),
        )

        args = dict(
            policy=self.policy_type,
            env=self.env,
            device=device,
            verbose=1,
            buffer_size=min(len(obs), 20000),
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
        )

        assert algo in ["DDPG", "TD3", "SAC"], "Unsupported algorithm"
        self.model: OffPolicyAlgorithm = eval(f"{algo}")(**args)

        # Setup checkpoint + best-model save
        self.checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix=f"{algo}_offline_model",
            verbose=1
        )

        self.eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(log_dir, "best_model"),
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=self.save_freq,
            deterministic=True,
            render=False
        )

        # Combine both
        self.callbacks = CallbackList([
            self.checkpoint_callback,
            self.eval_callback
        ])

    def train(self, **kwargs):
        kwargs.setdefault("total_timesteps", 1_000_000)

        # Pass callbacks
        self.model.learn(
            **kwargs,
            callback=self.callbacks
        )

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = self.model.load(path, env=self.vec_env)


class OfflineReplayEnv(gym.Env):
    """
    Offline replay environment:
      - Sequential cursor through offline dataset
      - Does NOT reset on dataset's done=True
      - Wraps around at end of dataset (ring buffer)
      - reset() starts at a random index (for multi-env)
    """

    metadata = {"render.modes": []}

    def __init__(self, obs, actions, rewards, next_obs, dones,
                 obs_space: spaces.Space, action_space: spaces.Space):

        super().__init__()

        self.obs_arr = obs
        self.actions_arr = actions
        self.rewards_arr = rewards
        self.next_obs_arr = next_obs
        self.dones_arr = dones

        self.observation_space = obs_space
        self.action_space = action_space

        self.size = len(obs)
        self.ptr = 0

    # Reset: choose a *random* starting index (important for venv)
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.ptr = np.random.randint(0, self.size)
        obs = self.obs_arr[self.ptr]
        info = {}

        return obs, info

    # Step: move pointer sequentially, wrap at end
    def step(self, action):
        i = self.ptr

        # Transition
        obs_next = self.next_obs_arr[i]
        reward = float(self.rewards_arr[i])
        terminated = bool(self.dones_arr[i])
        truncated = False

        # Increment ptr (wrap around)
        self.ptr = (self.ptr + 1) % self.size

        return obs_next, reward, terminated, truncated, {}
    