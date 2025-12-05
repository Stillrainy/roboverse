import numpy as np
import torch
from torch.utils.data import Dataset


class SkillSequenceDataset(Dataset):
    """
    Random variable-length skill sequence sampler.

    Parameters:
    - obs: list of dicts, len = N
    - act: np.ndarray (N, A)
    - dones: np.ndarray (N,)
    - max_seq_len: maximum possible skill length (H_max)

    Sampling:
    - Random start index s
    - Compute max allowed H due to episode boundary
    - Sample a random H uniformly from {1, ..., H_allowed, ..., max_seq_len}
    - Return sequence a[s : s+H], obs[s], and H
    """

    def __init__(self, obs, act, dones, max_seq_len):
        super().__init__()
        self.obs = obs
        self.act = act
        self.dones = dones.astype(bool)
        self.max_seq_len = max_seq_len

        N = len(obs)
        assert act.shape[0] == N
        assert self.dones.shape[0] == N

        # collect valid start indices = any index where at least length 1 is valid
        self.start_indices = []
        for i in range(N):
            # cannot start on a terminal step
            if not self.dones[i]:
                self.start_indices.append(i)

        if len(self.start_indices) == 0:
            raise ValueError(
                "No valid start indices found (all are terminal states).")

        self.N = N

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        s = self.start_indices[idx]

        # Determine how long we can go before hitting next done=True
        # At least length 1 is always valid because start itself is non-terminal.
        max_len_until_done = 1
        limit = min(self.max_seq_len, self.N - s)

        for k in range(1, limit):
            if self.dones[s + k]:
                break
            max_len_until_done += 1

        # H ~ Uniform[1, max_len_until_done]
        H = np.random.randint(1, max_len_until_done + 1)

        # slice actions
        actions = self.act[s: s + H]      # (H, A)

        # first observation is the skill's conditioning obs
        o = self.obs[s]
        state = o['state']                  # (S,)
        image = o['image']                  # (C,H,W)
        depth = o['depth']                  # (H,W)

        # convert to tensors
        actions = torch.from_numpy(actions).float()           # (H, A)
        state = torch.from_numpy(state).float()             # (S,)
        image = torch.from_numpy(image).float() / 255.0     # (C,H,W)
        depth = torch.from_numpy(depth).unsqueeze(0).float()  # (1,H,W)

        return {
            'actions': actions,     # (H, A)  variable-length
            'state':   state,       # (S,)
            'image':   image,       # (C,H,W)
            'depth':   depth,       # (1,H,W)
            'H':       H            # actual sequence length
        }


def skill_collate_fn(batch):
    """
    batch: list of dicts from SkillSequenceDataset.__getitem__.

    Each element:
        'actions': (H_i, A)
        'state':   (S,)
        'image':   (C,H,W)
        'depth':   (1,H,W)
        'H':       int

    Returns:
        actions:       (B, H_max, A) padded with zeros
        actions_mask:  (B, H_max) bool, True for valid timesteps
        H:             (B,) long, original lengths
        state:         (B, S)
        image:         (B, C, H, W)
        depth:         (B, 1, H, W)
    """
    B = len(batch)
    # sequence lengths
    lengths = [item['actions'].shape[0] for item in batch]
    H_max = max(lengths)
    action_dim = batch[0]['actions'].shape[-1]

    # padded actions + mask
    actions = torch.zeros(B, H_max, action_dim,
                          dtype=batch[0]['actions'].dtype)
    actions_mask = torch.zeros(B, H_max, dtype=torch.bool)

    for i, item in enumerate(batch):
        H_i = item['actions'].shape[0]
        actions[i, :H_i] = item['actions']
        actions_mask[i, :H_i] = True

    state = torch.stack([item['state'] for item in batch], dim=0)
    image = torch.stack([item['image'] for item in batch], dim=0)
    depth = torch.stack([item['depth'] for item in batch], dim=0)
    H = torch.tensor(lengths, dtype=torch.long)

    return {
        'actions': actions,          # (B, H_max, A)
        'actions_mask': actions_mask,  # (B, H_max)
        'H': H,                      # (B,)
        'state': state,              # (B, S)
        'image': image,              # (B, C, H, W)
        'depth': depth,              # (B, 1, H, W)
    }
