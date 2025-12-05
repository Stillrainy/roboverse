import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

import roboverse
from roboverse.priors.PriorModels import (
    SkillPriorModel,
    kl_divergence_diag_gaussians
)
from roboverse.agents.utils import (
    SubspaceEnvWrapper,
    encode_obs_feat,
)
from roboverse.agents.skillprior_rls import (
    SkillPolicy,
    QNetwork,
    ReplayBuffer,
)


if __name__ == '__main__':
    env_name = 'Widow250DrawerRandomizedPickPlace-v0'
    prior_ckpt_path = './checkpoints/prior_latest.pt'
    total_env_steps = 500_000
    skill_horizon = 50
    replay_size = 100_000
    batch_size = 256
    gamma = 0.99
    lr = 3e-4
    start_steps = 1000
    update_after = 1000
    update_every = 50
    target_entropy = None
    target_divergence = 5.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- Load env ----------
    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    env = SubspaceEnvWrapper(
        env, keys=['state', 'image', 'depth'])
    eval_env = SubspaceEnvWrapper(
        eval_env, keys=['state', 'image', 'depth'])

    # ---------- Load prior model ----------
    ckpt = torch.load(prior_ckpt_path, map_location=device)
    config = ckpt['config']

    state_dim = config['state_dim']
    img_shape = tuple(config['img_shape'])
    depth_shape = tuple(config['depth_shape'])
    z_dim = config['z_dim']

    prior_model = SkillPriorModel(
        state_dim=state_dim,
        img_shape=img_shape,
        depth_shape=depth_shape,
        action_dim=env.action_space.shape[0],
        z_dim=z_dim,
    ).to(device)
    prior_model.load_state_dict(ckpt['model_state_dict'])
    prior_model.eval()
    for p in prior_model.parameters():
        p.requires_grad = False

    # Infer obs_feat_dim from prior
    with torch.no_grad():
        dummy_obs = env.reset()
        if isinstance(dummy_obs, tuple):  # gymnasium compatibility
            dummy_obs, _ = dummy_obs
        dummy_feat = encode_obs_feat(prior_model, dummy_obs, device)
        obs_feat_dim = dummy_feat.shape[0]

    # ---------- RL networks ----------
    policy = SkillPolicy(obs_feat_dim, z_dim).to(device)
    # initialize policy from prior (Section F in paper)
    with torch.no_grad():
        # copy prior.skill_prior's base MLP into policy.net where possible
        if hasattr(prior_model, 'skill_prior'):
            prior_mlp = prior_model.skill_prior.net  # Sequential
            pol_layers = [m for m in policy.net if isinstance(m, nn.Linear)]
            prior_layers = [m for m in prior_mlp if isinstance(m, nn.Linear)]
            for pl, pr in zip(pol_layers, prior_layers):
                pl.weight.data.copy_(pr.weight.data)
                pl.bias.data.copy_(pr.bias.data)

    q1 = QNetwork(obs_feat_dim, z_dim).to(device)
    q2 = QNetwork(obs_feat_dim, z_dim).to(device)
    q1_target = QNetwork(obs_feat_dim, z_dim).to(device)
    q2_target = QNetwork(obs_feat_dim, z_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    q1_optimizer = torch.optim.Adam(q1.parameters(), lr=lr)
    q2_optimizer = torch.optim.Adam(q2.parameters(), lr=lr)

    # temperature α for KL regularization, with target divergence δ
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr)

    def alpha(): return log_alpha.exp()

    # Replay buffer over obs_feat, z
    rb = ReplayBuffer(replay_size, obs_feat_dim, z_dim)

    def get_prior_dist(obs_feat_batch):
        """
        obs_feat_batch: (B, obs_feat_dim)
        returns mu_p, log_std_p
        """
        with torch.no_grad():
            # feed through same skill_prior head as in prior training
            mu_p, log_std_p = prior_model.skill_prior(obs_feat_batch)
        return mu_p, log_std_p

    def evaluate_policy(n_episodes=5):
        returns = []
        for _ in range(n_episodes):
            obs = eval_env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs
            obs_feat = encode_obs_feat(prior_model, obs, device)
            done = False
            ep_ret = 0.0
            steps = 0
            while not done:
                with torch.no_grad():
                    obs_feat_batch = obs_feat.unsqueeze(0)
                    mu, log_std = policy(obs_feat_batch)
                    z = mu  # deterministic eval (mean)
                    # decode into primitive actions
                    a_seq = prior_model.decode_skill(
                        z, seq_len=skill_horizon)  # (1,H,A)
                a_seq_np = a_seq.squeeze(0).cpu().numpy()  # (H,A)
                for a in a_seq_np:
                    next_obs, r, terminated, truncated, info = eval_env.step(a)
                    done = terminated or truncated
                    ep_ret += float(r)
                    steps += 1
                    if done:
                        break
                if not done:
                    obs_feat = encode_obs_feat(prior_model, next_obs, device)
            returns.append(ep_ret)
        return np.mean(returns), np.std(returns)

    # ---------- Main training loop ----------
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs
    obs_feat = encode_obs_feat(prior_model, obs, device)
    episode_return = 0.0
    episode_len = 0

    total_steps = 0
    target_update_tau = 5e-3

    while total_steps < total_env_steps:
        # --- Sample skill z from policy or random warmup ---
        if total_steps < start_steps:
            # random z for exploration
            z = torch.randn(1, z_dim, device=device)
        else:
            with torch.no_grad():
                obs_feat_batch = obs_feat.unsqueeze(0)  # (1,obs_feat_dim)
                z, _, _ = policy.sample(obs_feat_batch)  # (1,z_dim)

        # decode into primitive actions via prior decoder
        with torch.no_grad():
            a_seq = prior_model.decode_skill(
                z, seq_len=skill_horizon)  # (1,H,A)
        a_seq_np = a_seq.squeeze(0).cpu().numpy()  # (H,A)

        # --- Execute skill in environment for H steps ---
        total_reward = 0.0
        terminated = False
        truncated = False
        for h in range(skill_horizon):
            a = a_seq_np[h]
            next_obs, r, term, trunc, info = env.step(a)
            total_reward += float(r)
            total_steps += 1
            episode_len += 1
            done = term or trunc

            if done or total_steps >= total_env_steps:
                terminated, truncated = term, trunc
                break

        next_obs_dict = next_obs
        next_obs_feat = encode_obs_feat(prior_model, next_obs_dict, device)

        # store high-level transition (obs_feat, z, R̃, next_obs_feat, done)
        rb.add(
            obs_feat.cpu().numpy(),
            z.squeeze(0).detach().cpu().numpy(),
            total_reward,
            next_obs_feat.cpu().numpy(),
            float(done),
        )

        obs_feat = next_obs_feat
        episode_return += total_reward

        if done or total_steps >= total_env_steps:
            print(
                f"[Step {total_steps}] Episode return: {episode_return:.2f}, len: {episode_len}")
            obs = env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs
            obs_feat = encode_obs_feat(prior_model, obs, device)
            episode_return = 0.0
            episode_len = 0

        # --- SAC updates ---
        if total_steps >= update_after and total_steps % update_every == 0 and rb.size >= batch_size:
            for _ in range(update_every):
                obs_batch, z_batch, r_batch, next_obs_batch, done_batch = rb.sample(
                    batch_size)

                # Policy output at next state
                with torch.no_grad():
                    mu_next, log_std_next = policy(next_obs_batch)
                    std_next = log_std_next.exp()
                    eps_next = torch.randn_like(std_next)
                    z_next = mu_next + eps_next * std_next  # (B,z_dim)

                    q1_next = q1_target(next_obs_batch, z_next)
                    q2_next = q2_target(next_obs_batch, z_next)
                    q_next = torch.min(q1_next, q2_next)

                    # KL(π(.|s') || p_a(.|s'))
                    mu_p_next, log_std_p_next = get_prior_dist(next_obs_batch)
                    kl_next = kl_divergence_diag_gaussians(
                        mu_next,
                        log_std_next,
                        mu_p_next,
                        log_std_p_next)

                    # Q target: r + γ ( Q - α KL )
                    y = r_batch + gamma * \
                        (1.0 - done_batch) * (q_next - alpha() * kl_next)

                # Critic loss
                q1_val = q1(obs_batch, z_batch)
                q2_val = q2(obs_batch, z_batch)
                q1_loss = F.mse_loss(q1_val, y)
                q2_loss = F.mse_loss(q2_val, y)

                q1_optimizer.zero_grad()
                q1_loss.backward()
                q1_optimizer.step()

                q2_optimizer.zero_grad()
                q2_loss.backward()
                q2_optimizer.step()

                # Policy loss
                mu, log_std = policy(obs_batch)
                std = log_std.exp()
                eps = torch.randn_like(std)
                z_sample = mu + eps * std

                q1_pi = q1(obs_batch, z_sample)
                q2_pi = q2(obs_batch, z_sample)
                q_pi = torch.min(q1_pi, q2_pi)

                mu_p, log_std_p = get_prior_dist(obs_batch)
                kl_pi = kl_divergence_diag_gaussians(
                    mu,
                    log_std,
                    mu_p,
                    log_std_p
                )

                policy_loss = -(q_pi - alpha() * kl_pi).mean()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Temperature (alpha) update for target divergence δ (Alg. 1, line 12)
                alpha_loss = (
                    alpha() * (kl_pi.detach() - target_divergence)).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                # Soft target update
                for param, target_param in zip(q1.parameters(), q1_target.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - target_update_tau) +
                        param.data * target_update_tau
                    )
                for param, target_param in zip(q2.parameters(), q2_target.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - target_update_tau) +
                        param.data * target_update_tau
                    )

            # Optional: evaluation
            eval_mean, eval_std = evaluate_policy(n_episodes=3)
            print(f"[Step {total_steps}] Eval return: {eval_mean:.2f} ± {eval_std:.2f}, "
                  f"alpha={alpha().item():.3f}")
