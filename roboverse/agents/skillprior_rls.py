import numpy as np
import torch
import torch.nn as nn


# ---------- Replay Buffer over skills ----------

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, z_dim, device):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.z = np.zeros((capacity, z_dim), dtype=np.float32)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(self, obs, z, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.z[self.ptr] = z
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idxs]).to(self.device),
            torch.from_numpy(self.z[idxs]).to(self.device),
            torch.from_numpy(self.rew[idxs]).to(self.device),
            torch.from_numpy(self.next_obs[idxs]).to(self.device),
            torch.from_numpy(self.done[idxs]).to(self.device),
        )


# ---------- Networks: High-level policy and critics ----------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SkillPolicy(nn.Module):
    """
    High-level policy πθ(z | s): Gaussian over z.
    We'll initialize it from the skill prior later (same arch).
    """
    def __init__(self, obs_feat_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_feat_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(128, z_dim)
        self.fc_logstd = nn.Linear(128, z_dim)

    def forward(self, obs_feat):
        h = self.net(obs_feat)
        mu = self.fc_mu(h)
        log_std = self.fc_logstd(h).clamp(-5.0, 2.0)
        return mu, log_std

    def sample(self, obs_feat):
        mu, log_std = self.forward(obs_feat)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, log_std


class QNetwork(nn.Module):
    """
    Qφ(s, z): critic. Input is concatenated (obs_feat, z).
    """
    def __init__(self, obs_feat_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_feat_dim + z_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs_feat, z):
        x = torch.cat([obs_feat, z], dim=-1)
        return self.net(x).squeeze(-1)