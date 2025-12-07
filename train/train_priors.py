import argparse
import os
from datetime import datetime
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from roboverse.datasets.PriorDataset import (
    SkillSequenceDataset,
    skill_collate_fn
)
from roboverse.priors.PriorModels import SkillPriorModel
from roboverse.agents.utils import load_dataset_from_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train skill prior model')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing dataset')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='Max skill sequence length (H_max)')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--z_dim', type=int, default=10,
                        help='Latent dimension for skill prior')
    parser.add_argument('--beta', type=float, default=1e-2,
                        help='KL weight beta')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    default_ckpt = os.path.join(
        'logs',
        'prior',
        datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--ckpt_dir', default=default_ckpt,
                        help='Checkpoint output directory')
    args = parser.parse_args()

    # Hyperparameters
    device = args.device
    max_seq_len = args.max_seq_len
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    z_dim = args.z_dim
    beta = args.beta

    def _scalar(val):
        return val.item() if torch.is_tensor(val) else float(val)

    # Directories
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'tb'))

    # Save hyperparameters
    with open(os.path.join(ckpt_dir, 'hyperparams.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

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

    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)

    # Training loop
    global_step = 0
    epoch_bar = tqdm(range(num_epochs), desc='Epochs', unit='epoch')
    for epoch in epoch_bar:
        model.train()
        running = {'total_loss': 0.0, 'recon_loss': 0.0,
                   'kl_to_unit': 0.0, 'kl_to_prior': 0.0}
        n_batches = 0

        batch_bar = tqdm(loader, desc=f'Epoch {epoch:03d}', leave=False)
        for batch in batch_bar:
            for k in batch:
                batch[k] = batch[k].to(device)

            optimizer.zero_grad()
            out = model(batch)
            loss, terms = model.loss(out, beta=beta)
            loss.backward()
            optimizer.step()

            # accumulate running means
            for k in running:
                running[k] += _scalar(terms.get(k, 0.0))
            n_batches += 1

            # per-batch tensorboard logging
            batch_total = _scalar(terms.get('total_loss', loss))
            batch_recon = _scalar(terms.get('recon_loss', 0.0))
            batch_kl_unit = _scalar(terms.get('kl_to_unit', 0.0))
            batch_kl_prior = _scalar(terms.get('kl_to_prior', 0.0))
            writer.add_scalar('batch/total_loss', batch_total, global_step)
            writer.add_scalar('batch/recon_loss', batch_recon, global_step)
            writer.add_scalar('batch/kl_to_unit', batch_kl_unit, global_step)
            writer.add_scalar('batch/kl_to_prior', batch_kl_prior, global_step)
            global_step += 1

        for k in running:
            running[k] /= max(n_batches, 1)

        # per-epoch tensorboard logging and progress bar update
        writer.add_scalar('epoch/total_loss', running['total_loss'], epoch)
        writer.add_scalar('epoch/recon_loss', running['recon_loss'], epoch)
        writer.add_scalar('epoch/kl_to_unit', running['kl_to_unit'], epoch)
        writer.add_scalar('epoch/kl_to_prior', running['kl_to_prior'], epoch)
        epoch_bar.set_postfix({'loss': f"{running['total_loss']:.4f}"})

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
    writer.close()
