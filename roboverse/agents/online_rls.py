import os
import gymnasium as gym
from typing import List
from stable_baselines3 import DDPG, TD3, SAC
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
    ImageDepthEnvWrapper,
    SubspaceEnvWrapper
)
from .custom_cnn import CustomCNN


class OnlineAgentTrainer:
    def __init__(
        self,
        env_id: str,
        obs_keys: List[str] = None,
        algo: str = "SAC",
        device: str = "cuda",
        log_dir: str = "./logs_online",
        save_freq: int = 10_000,
    ):

        self.env_id = env_id
        self.obs_keys = obs_keys
        self.log_dir = log_dir
        self.save_freq = save_freq

        # Create environment
        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)

        # Wrap env if needed
        if obs_keys is not None and any(k in ["image", "depth"] for k in obs_keys):
            self.env = ImageDepthEnvWrapper(self.env)
            self.eval_env = ImageDepthEnvWrapper(self.eval_env)
        elif obs_keys is not None:
            self.env = SubspaceEnvWrapper(self.env, obs_keys)
            self.eval_env = SubspaceEnvWrapper(self.eval_env, obs_keys)

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
            buffer_size=20000,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
        )

        assert algo in ["DDPG", "TD3", "SAC"]
        self.model: OffPolicyAlgorithm = eval(f"{algo}")(**args)

        # Callbacks (Checkpoint + Eval)
        self.checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix=f"{algo}_online_model"
        )

        # Evaluation environment
        self.eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(log_dir, "best_model"),
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=save_freq,
            deterministic=True,
        )

        self.callbacks = CallbackList([
            self.checkpoint_callback,
            self.eval_callback
        ])

    # Training
    def train(self, **kwargs):
        kwargs.setdefault("total_timesteps", 1_000_000)
        self.model.learn(
            **kwargs,
            callback=self.callbacks
        )

    # Save / Load model
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = self.model.load(path, env=self.vec_env)
