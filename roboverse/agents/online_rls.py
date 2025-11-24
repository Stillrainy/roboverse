import os
import gymnasium as gym
from typing import List
from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)

from .utils import ImageDepthEnvWrapper
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
        n_envs: int = 1,
    ):

        self.env_id = env_id
        self.obs_keys = obs_keys
        self.log_dir = log_dir
        self.save_freq = save_freq

        # Create environment(s)
        def make_env(rank):
            def _thunk():
                env = gym.make(env_id)

                # If observation space is dict and user selects modal keys
                if isinstance(env.observation_space, gym.spaces.Dict):
                    if obs_keys is not None:
                        # Filter only selected keys
                        sub_spaces = {k: env.observation_space[k] for k in obs_keys}
                        env.observation_space = gym.spaces.Dict(sub_spaces)

                    # Wrap with image-depth wrapper if needed
                    if obs_keys and any(k in ["image", "depth"] for k in obs_keys):
                        env = ImageDepthEnvWrapper(env)

                return env
            return _thunk

        # Vectorized env
        if n_envs == 1:
            self.vec_env = DummyVecEnv([make_env(0)])
        else:
            self.vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])


        # Auto-select policy type (Cnn vs Mlp)
        obs_space = self.vec_env.observation_space
        if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3:
            # (C,H,W) tensor → image
            self.policy_type = "CnnPolicy"
            normalize_images = False
        else:
            self.policy_type = "MlpPolicy"
            normalize_images = True

        # SB3 Policy kwargs
        policy_kwargs = dict(
            normalize_images=normalize_images,
            # features_extractor_class=CustomCNN,
            # features_extractor_kwargs=dict(features_dim=256),
        )

        # SB3 Model
        tb_log_path = os.path.join(log_dir, "tb")

        args = dict(
            policy=self.policy_type,
            env=self.vec_env,
            device=device,
            verbose=1,
            tensorboard_log=tb_log_path,
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

        # Evaluation environment (always 1 env)
        self.eval_env = DummyVecEnv([make_env(12345)])
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
