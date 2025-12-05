import os
import torch
from torch.utils.data import DataLoader

from roboverse.datasets.PriorDataset import (
    SkillSequenceDataset,
    skill_collate_fn
)
from roboverse.priors.PriorModels import SkillPriorModel
from roboverse.agents.utils import load_dataset_from_dir


if __name__ == '__main__':
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_seq_len = 50          # used as H_max for skill training
    batch_size = 64
    num_epochs = 50
    lr = 1e-3
    z_dim = 10
    beta = 1e-2

    # Directories
    data_dir = 'data/data_Widow250DrawerRandomizedPickPlace-v0_100_noise_0.1_2025-11-23T17-17-43'
    ckpt_dir = './checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load dataset
    obs, act, rew, next_obs, dones = load_dataset_from_dir(
        data_dir, ['state', 'image', 'depth']
    )

    state_dim = obs[0]['state'].shape[0]
    img_shape = obs[0]['image'].shape      # (C,H,W)
    depth_shape = obs[0]['depth'].shape    # (H,W)

    dataset = SkillSequenceDataset(
        obs=obs,
        act=act,
        dones=dones,
        max_seq_len=max_seq_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=skill_collate_fn,
    )

    # Initialize model
    model = SkillPriorModel(
        state_dim=state_dim,
        img_shape=img_shape,
        depth_shape=depth_shape,
        action_dim=act.shape[1],
        z_dim=z_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running = {'total_loss': 0.0, 'recon_loss': 0.0,
                   'kl_to_unit': 0.0, 'kl_to_prior': 0.0}
        n_batches = 0

        for batch in loader:
            # move to device
            for k in batch:
                batch[k] = batch[k].to(device)

            optimizer.zero_grad()
            out = model(batch)
            loss, terms = model.loss(out, beta=beta)
            loss.backward()
            optimizer.step()

            for k in running:
                running[k] += terms.get(k, 0.0)
            n_batches += 1

        for k in running:
            running[k] /= max(n_batches, 1)
        print(f"Epoch {epoch:03d}: "
              f"total={running['total_loss']:.4f}, "
              f"recon={running['recon_loss']:.4f}, "
              f"kl_unit={running['kl_to_unit']:.4f}, "
              f"kl_prior={running['kl_to_prior']:.4f}")

    # --- Save final/latest checkpoint ---
    latest_path = os.path.join(ckpt_dir, 'prior_latest.pt')
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'max_seq_len': max_seq_len,
            'z_dim': z_dim,
            'beta': beta,
            'state_dim': state_dim,
            'img_shape': img_shape,
            'depth_shape': depth_shape,
        }
    }, latest_path)
    print(f"Saved final prior model to {latest_path}")
