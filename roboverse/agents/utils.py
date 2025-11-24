import numpy as np
from typing import List, Dict, Any, Tuple
import gymnasium as gym
from gymnasium import spaces


def load_dataset_file(path: str, obs_keys: List[str] = None):
    """
    Dataset format:
        data = [
            {
                "observations":         [ {obs_key:...}, {obs_key:...}, ... ],
                "next_observations":    [ {obs_key:...}, ... ],
                "actions":              [...],
                "rewards":              [...],
                "terminals":            [...],
            },
            ...
        ]

    obs_keys:
        - None → load ALL observation keys
        - ['image', 'depth'] → load selected keys only

    Returns:
        obs:       array(dtype=object) of dicts
        next_obs:  array(dtype=object) of dicts
        actions:   np.array
        rewards:   np.array
        dones:     np.array
    """

    data = np.load(path, allow_pickle=True)

    obs_list = []
    next_obs_list = []
    act_list = []
    rew_list = []
    done_list = []

    # ------------------------------------------------------------
    # Loop through each trajectory
    # ------------------------------------------------------------
    for traj in data:

        traj_obs = traj["observations"]             # list of dicts
        traj_next_obs = traj["next_observations"]   # list of dicts
        traj_actions = traj["actions"]
        traj_rewards = traj["rewards"]
        traj_dones = traj["terminals"]

        T = len(traj_obs)

        for t in range(T):

            obs_raw = traj_obs[t]            # dict
            next_obs_raw = traj_next_obs[t]  # dict

            # -----------------------------------------
            # If user didn't specify obs_keys → use all
            # -----------------------------------------
            if obs_keys is None:
                keys = list(obs_raw.keys())
            else:
                keys = obs_keys

            # Extract subset of keys into a NEW dict
            obs_entry = {k: obs_raw[k] for k in keys}
            next_obs_entry = {k: next_obs_raw[k] for k in keys}

            obs_list.append(obs_entry)
            next_obs_list.append(next_obs_entry)
            act_list.append(traj_actions[t])
            rew_list.append(traj_rewards[t])
            done_list.append(traj_dones[t])

    # ------------------------------------------------------------
    # Convert to numpy arrays (dtype=object)
    # ------------------------------------------------------------
    obs = np.array(obs_list, dtype=object)
    next_obs = np.array(next_obs_list, dtype=object)
    actions = np.array(act_list)
    rewards = np.array(rew_list)
    dones = np.array(done_list)

    return obs, actions, rewards, next_obs, dones


def load_dataset_from_dir(data_dir: str, obs_keys: List[str] = None):
    import os
    import glob
    file = sorted(glob.glob(os.path.join(data_dir, "*.npy")))[0]
    return load_dataset_file(file, obs_keys)


def build_space(env: gym.Env, obs_keys: List[str]) -> Tuple[spaces.Space, spaces.Space]:
    """
    Build observation and action spaces based on obs_keys.

    Args:
    env: gym environment
    obs_keys: list of keys to extract: ['image'], ['image','depth'], etc.

    Returns:
    obs_space: gym.spaces.Space for observations
    action_space: gym.spaces.Space for actions
    """
    obs_space = {}
    assert isinstance(env.observation_space,
                      spaces.Dict), "Environment observation space must be of type spaces.Dict"
    for k in obs_keys:
        if k not in env.observation_space.spaces:
            raise ValueError(f"Key {k} not in environment observation space.")
        else:
            obs_space[k] = env.observation_space.spaces[k]

    obs_space = spaces.Dict(obs_space)

    action_space = env.action_space

    return obs_space, action_space


class ImageDepthEnvWrapper(gym.ObservationWrapper):
    """
    Convert dict observation:
        {
            "image": uint8 (3, H, W),
            "depth": float32 (H, W)
        }
    Into normalized stacked (4, H, W) float32:
        - image normalized to [0,1]
        - depth normalized (div by depth_max) and expanded to (1,H,W)
    """

    def __init__(self, env, depth_max=1.0):
        super().__init__(env)

        self.depth_max = depth_max

        # Extract shape info from the env's first observation
        sample = env.reset()[0]

        img = sample["image"]        # uint8 (3,H,W)
        depth = sample["depth"]      # float (H,W)

        self.H = img.shape[1]
        self.W = img.shape[2]

        assert img.dtype == np.uint8, "image must be uint8"
        assert depth.ndim == 2, "depth must be (H,W)"

        # Build the new observation space (4,H,W)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4, self.H, self.W),
            dtype=np.float32
        )

    def observation(self, obs):
        """
        obs is dict:
            { "image": uint8 (3,H,W), "depth": float32 (H,W) }
        """

        img = obs["image"].astype(np.float32) / 255.0   # (3,H,W)
        depth = obs["depth"].astype(np.float32)         # (H,W)

        # Normalize depth, clip to [0,1]
        depth_norm = np.clip(depth / self.depth_max, 0.0, 1.0)
        depth_norm = depth_norm[None, ...]              # (1,H,W)

        # Stack into final observation (4,H,W)
        stacked = np.concatenate([img, depth_norm], axis=0)  # float32

        return stacked


class SubspaceEnvWrapper(gym.ObservationWrapper):
    """
    Extract a sub-dictionary from the observation dict.
    """

    def __init__(self, env, keys: List[str]):
        super().__init__(env)
        self.keys = keys

        # Build the new observation space
        obs_space = {}
        for k in keys:
            if k not in env.observation_space.spaces:
                raise ValueError(f"Key {k} not in environment observation space.")
            else:
                obs_space[k] = env.observation_space.spaces[k]

        self.observation_space = spaces.Dict(obs_space)

    def observation(self, obs):
        """
        obs is dict
        """
        sub_obs = {k: obs[k] for k in self.keys}
        return sub_obs
