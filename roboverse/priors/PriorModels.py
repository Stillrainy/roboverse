import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod


class SkillPriorModel(nn.Module):
    """
    Combines:
    - Skill encoder q(z|a_{1:H})
    - Skill decoder p(a_{1:H}|z)
    - Obs encoder f(o_t)
    - State-conditioned skill prior p_a(z|o_t)
    """

    def __init__(self, state_dim, img_shape, depth_shape,
                 action_dim, z_dim=10, obs_latent_dim=256, hidden_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim

        self.obs_encoder = ObsEncoder(
            state_dim=state_dim,
            img_shape=img_shape,
            depth_shape=depth_shape,
            obs_latent_dim=obs_latent_dim,
        )
        self.skill_encoder = SkillEncoder(
            action_dim=action_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
        )
        self.skill_decoder = SkillDecoder(
            action_dim=action_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
        )
        self.skill_prior = SkillPriorNet(
            obs_latent_dim=obs_latent_dim,
            z_dim=z_dim,
        )

    def encode_obs(self, state, image, depth):
        return self.obs_encoder(state, image, depth)

    def encode_skill(self, actions):
        return self.skill_encoder(actions)

    def decode_skill(self, z, seq_len):
        return self.skill_decoder(z, seq_len)

    def prior_dist(self, obs_feat):
        return self.skill_prior(obs_feat)

    def forward(self, batch):
        """
        batch: dict from SkillSequenceDataset
            actions: (B,H,A)
            state:   (B,S)
            image:   (B,C,H,W)
            depth:   (B,1,H,W)

        Returns dict of components.
        """
        actions = batch['actions']           # (B,H,A)
        state = batch['state']
        image = batch['image']
        depth = batch['depth']
        B, H, A = actions.shape

        obs_feat = self.encode_obs(
            state, image, depth)       # (B,obs_latent_dim)
        mu_z, log_std_z = self.encode_skill(
            actions)          # posterior q(z|a)
        std_z = torch.exp(log_std_z)
        # reparameterization trick
        eps = torch.randn_like(std_z)
        z = mu_z + eps * std_z                                # (B,z_dim)

        a_hat = self.decode_skill(z, seq_len=H)               # (B,H,A)

        mu_p, log_std_p = self.prior_dist(obs_feat)           # prior p_a(z|o)

        return {
            'actions': actions,
            'a_hat': a_hat,
            'mu_z': mu_z,
            'log_std_z': log_std_z,
            'mu_p': mu_p,
            'log_std_p': log_std_p,
        }

    @staticmethod
    def loss(output, beta=1e-2):
        """
        Compute SPiRL-style loss components.

        output is the dict from SkillPriorModel.forward().
        Returns total_loss, dict_of_terms
        """
        actions = output['actions']      # (B,H,A)
        a_hat = output['a_hat']         # (B,H,A)
        mu_z = output['mu_z']           # (B,z_dim)
        log_std_z = output['log_std_z'] # (B,z_dim)
        mu_p = output['mu_p']           # (B,z_dim)
        log_std_p = output['log_std_p'] # (B,z_dim)

        B = actions.size(0)

        # Reconstruction loss (MSE over sequence and actions)
        recon_loss = F.mse_loss(a_hat, actions, reduction='sum') / B

        # KL(q(z|a) || N(0, I))
        zero = torch.zeros_like(mu_z)
        kl_to_unit = kl_divergence_diag_gaussians(mu_z, log_std_z, zero, zero).mean()

        # KL(q(z|a) || p_a(z|o)), with stop-gradient on q
        mu_z_detached = mu_z.detach()
        log_std_z_detached = log_std_z.detach()
        kl_to_prior = kl_divergence_diag_gaussians(
            mu_z_detached, log_std_z_detached, mu_p, log_std_p
        ).mean()

        total_loss = recon_loss + beta * kl_to_unit + kl_to_prior

        return total_loss, {
            'recon_loss': recon_loss.item(),
            'kl_to_unit': kl_to_unit.item(),
            'kl_to_prior': kl_to_prior.item(),
            'total_loss': total_loss.item(),
        }


class SkillEncoder(nn.Module):
    """
    LSTM encoder over action sequences: q(z | a_{1:H})
    """

    def __init__(self, action_dim, z_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=action_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logstd = nn.Linear(hidden_dim, z_dim)

    def forward(self, actions):
        """
        actions: (B, H, A)
        returns: mu_z, log_std_z  each (B, z_dim)
        """
        _, (h_n, _) = self.lstm(actions)   # h_n: (1, B, hidden_dim)
        h = h_n.squeeze(0)                # (B, hidden_dim)
        mu = self.fc_mu(h)
        log_std = self.fc_logstd(h).clamp(min=-5.0, max=5.0)
        return mu, log_std


class SkillDecoder(nn.Module):
    """
    LSTM decoder that reconstructs action sequence from z.
    Condition on z at every step.
    """

    def __init__(self, action_dim, z_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = action_dim + z_dim
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, z, seq_len):
        """
        z: (B, z_dim)
        seq_len: H (int)
        returns: reconstructed actions hat_a: (B, H, A)
        """
        B, z_dim = z.shape
        # repeat z across time and concatenate with a zero action input
        zeros_a = torch.zeros(
            B, seq_len, self.fc_out.out_features, device=z.device)
        z_rep = z.unsqueeze(1).repeat(1, seq_len, 1)      # (B,H,z_dim)
        dec_in = torch.cat([zeros_a, z_rep], dim=-1)      # (B,H,A+z_dim)
        h, _ = self.lstm(dec_in)
        a_hat = self.fc_out(h)                            # (B,H,A)
        return a_hat


class ObsEncoder(nn.Module):
    """
    Encodes (state, image, depth) into a single feature vector.
    """

    def __init__(self, state_dim, img_shape, depth_shape, obs_latent_dim=256):
        super().__init__()
        C, H, W = img_shape
        # state MLP
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
        )

        # image CNN
        self.img_cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=2,
                      padding=1),  # (32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2,
                      padding=1),  # (64, H/4, W/4)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2,
                      padding=1),  # (64, H/8, W/8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        # depth CNN (1-channel input)
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        # compute flattened sizes
        with torch.no_grad():
            dummy_img = torch.zeros(1, C, H, W)
            dummy_depth = torch.zeros(1, 1, depth_shape[0], depth_shape[1])
            img_feat_dim = self.img_cnn(dummy_img).view(1, -1).shape[1]
            depth_feat_dim = self.depth_cnn(dummy_depth).view(1, -1).shape[1]
        self.img_feat_dim = img_feat_dim
        self.depth_feat_dim = depth_feat_dim

        total_in = 128 + img_feat_dim + depth_feat_dim
        self.fc = nn.Sequential(
            nn.Linear(total_in, obs_latent_dim),
            nn.LeakyReLU(),
            nn.Linear(obs_latent_dim, obs_latent_dim),
            nn.LeakyReLU(),
        )

    def forward(self, state, image, depth):
        """
        state: (B, S)
        image: (B, C, H, W)
        depth: (B, 1, H, W)
        """
        s_feat = self.state_mlp(state)                     # (B, 128)
        img_feat = self.img_cnn(image).view(image.size(0), -1)
        d_feat = self.depth_cnn(depth).view(depth.size(0), -1)
        x = torch.cat([s_feat, img_feat, d_feat], dim=-1)
        # (B, obs_latent_dim)
        return self.fc(x)


class SkillPriorNet(nn.Module):
    """
    Maps encoded observation features to a Gaussian prior over z: p_a(z | o_t)
    """

    def __init__(self, obs_latent_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_latent_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(128, z_dim)
        self.fc_logstd = nn.Linear(128, z_dim)

    def forward(self, obs_feat):
        """
        obs_feat: (B, obs_latent_dim)
        """
        h = self.net(obs_feat)
        mu = self.fc_mu(h)
        log_std = self.fc_logstd(h).clamp(min=-5.0, max=5.0)
        return mu, log_std


def kl_divergence_diag_gaussians(mu_q, log_std_q, mu_p, log_std_p):
    """
    KL( N(mu_q, sigma_q^2 I) || N(mu_p, sigma_p^2 I) ) for diagonal Gaussians.
    mu_*, log_std_*: (B, D)
    """
    std_q = torch.exp(log_std_q)
    std_p = torch.exp(log_std_p)

    var_q = std_q ** 2
    var_p = std_p ** 2

    # KL = log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2)/(2 sigma_p^2) - 1/2
    kl = (log_std_p - log_std_q) + \
        (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p) - 0.5
    return kl.sum(dim=-1)  # (B,)
