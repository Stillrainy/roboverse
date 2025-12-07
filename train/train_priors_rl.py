import argparse
import os
from datetime import datetime
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(
        description='Train RL policy with skill prior')
    parser.add_argument('--env_name', default='Widow250DrawerRandomizedPickPlace-v0',
                        help='Environment name')
    parser.add_argument('--prior_ckpt_path', required=True,
                        help='Path to trained prior checkpoint')
    parser.add_argument('--total_env_steps', type=int, default=500_000,
                        help='Total environment steps')
    parser.add_argument('--skill_horizon', type=int, default=50,
                        help='Skill horizon (H)')
    parser.add_argument('--replay_size', type=int, default=10_000,
                        help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for updates')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--start_steps', type=int, default=1000,
                        help='Steps of random skills for exploration')
    parser.add_argument('--update_after', type=int, default=1000,
                        help='Steps before starting updates')
    parser.add_argument('--update_every', type=int, default=1,
                        help='Episodes between update batches')
    parser.add_argument('--target_entropy', type=float, default=None,
                        help='Optional target entropy (unused placeholder)')
    parser.add_argument('--target_divergence', type=float, default=5.0,
                        help='Target KL divergence delta')
    default_log_dir = os.path.join(
        'logs',
        'prior_rl',
        datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--log_dir', default=default_log_dir,
                        help='TensorBoard log directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    args = parser.parse_args()

    env_name = args.env_name
    prior_ckpt_path = args.prior_ckpt_path
    total_env_steps = args.total_env_steps
    skill_horizon = args.skill_horizon
    replay_size = args.replay_size
    batch_size = args.batch_size
    gamma = args.gamma
    lr = args.lr
    start_steps = args.start_steps
    update_after = args.update_after
    update_every = args.update_every
    target_entropy = args.target_entropy
    target_divergence = args.target_divergence
    device = args.device

    def _scalar(x):
        return x.item() if torch.is_tensor(x) else float(x)

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tb'))

    # Save hyperparameters
    with open(os.path.join(args.log_dir, 'hyperparams.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

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
    rb = ReplayBuffer(replay_size, obs_feat_dim, z_dim, device)

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
    episode_count = 0
    target_update_tau = 5e-3
    pbar = tqdm(total=total_env_steps, desc='Env steps', unit='step')
    last_loss = 0.0

    best_eval_mean = -float('inf')
    best_model_dir = os.path.join(args.log_dir, 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)

    while total_steps < total_env_steps:
        # --- Sample skill z from policy or random warmup ---
        if total_steps < start_steps:
            z = torch.randn(1, z_dim, device=device)
        else:
            with torch.no_grad():
                obs_feat_batch = obs_feat.unsqueeze(0)
                z, _, _ = policy.sample(obs_feat_batch)

        # decode into primitive actions via prior decoder
        with torch.no_grad():
            a_seq = prior_model.decode_skill(z, seq_len=skill_horizon)
        a_seq_np = a_seq.squeeze(0).cpu().numpy()

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
            pbar.update(1)

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
            episode_count += 1
            writer.add_scalar('train/episode_return',
                              episode_return, total_steps)
            writer.add_scalar('train/episode_length', episode_len, total_steps)
            obs = env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs
            obs_feat = encode_obs_feat(prior_model, obs, device)
            episode_return = 0.0
            episode_len = 0

        # --- SAC updates ---
        if total_steps >= update_after and episode_count % update_every == 0 and rb.size >= batch_size and done:
            for _ in range(update_every):
                obs_batch, z_batch, r_batch, next_obs_batch, done_batch = rb.sample(
                    batch_size)

                # Policy output at next state
                with torch.no_grad():
                    mu_next, log_std_next = policy(next_obs_batch)
                    std_next = log_std_next.exp()
                    eps_next = torch.randn_like(std_next)
                    z_next = mu_next + eps_next * std_next

                    q1_next = q1_target(next_obs_batch, z_next)
                    q2_next = q2_target(next_obs_batch, z_next)
                    q_next = torch.min(q1_next, q2_next)

                    mu_p_next, log_std_p_next = get_prior_dist(next_obs_batch)
                    kl_next = kl_divergence_diag_gaussians(
                        mu_next,
                        log_std_next,
                        mu_p_next,
                        log_std_p_next)

                    y = r_batch + gamma * \
                        (1.0 - done_batch) * (q_next - alpha() * kl_next)

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

                alpha_loss = (
                    alpha() * (kl_pi.detach() - target_divergence)).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                for param, target_param in zip(q1.parameters(), q1_target.parameters()):
                    target_param.data.copy_(
                        target_param.data *
                        (1.0 - target_update_tau) +
                        param.data * target_update_tau
                    )
                for param, target_param in zip(q2.parameters(), q2_target.parameters()):
                    target_param.data.copy_(
                        target_param.data *
                        (1.0 - target_update_tau) +
                        param.data * target_update_tau
                    )

                writer.add_scalar(
                    'train/q1_loss', _scalar(q1_loss), total_steps)
                writer.add_scalar(
                    'train/q2_loss', _scalar(q2_loss), total_steps)
                writer.add_scalar('train/policy_loss',
                                  _scalar(policy_loss), total_steps)
                writer.add_scalar('train/alpha', _scalar(alpha()), total_steps)
                writer.add_scalar('train/alpha_loss',
                                  _scalar(alpha_loss), total_steps)
                writer.add_scalar('train/kl_pi_mean',
                                  _scalar(kl_pi.mean()), total_steps)
                last_loss = _scalar(q1_loss + q2_loss + policy_loss)

            eval_mean, eval_std = evaluate_policy(n_episodes=3)
            writer.add_scalar('eval/return_mean', eval_mean, total_steps)
            writer.add_scalar('eval/return_std', eval_std, total_steps)
            writer.add_scalar('eval/alpha', _scalar(alpha()), total_steps)
            pbar.set_postfix({'loss': f"{last_loss:.4f}"})

            if eval_mean > best_eval_mean:
                best_eval_mean = eval_mean
                # Save best model
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'q1_state_dict': q1.state_dict(),
                    'q2_state_dict': q2.state_dict(),
                    'q1_target_state_dict': q1_target.state_dict(),
                    'q2_target_state_dict': q2_target.state_dict(),
                    'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                    'q1_optimizer_state_dict': q1_optimizer.state_dict(),
                    'q2_optimizer_state_dict': q2_optimizer.state_dict(),
                    'alpha': log_alpha.detach().cpu(),
                    'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
                    'config': {
                        'obs_feat_dim': obs_feat_dim,
                        'z_dim': z_dim,
                    },
                    'eval_return': eval_mean,
                    'eval_std': eval_std,
                    'total_steps': total_steps,
                }, os.path.join(best_model_dir, 'model.pt'))
                print(f"\nNew best eval return: {eval_mean:.4f} (std: {eval_std:.4f}) at step {total_steps}")

    pbar.close()
    writer.close()
