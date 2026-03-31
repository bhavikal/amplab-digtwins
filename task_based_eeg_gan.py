"""
Task-Based EEG GAN Training for HBN Dataset

This module implements Wasserstein GAN (WGAN) training for task-based EEG data,
enabling synthesis and analysis of synthetic EEG for different cognitive tasks.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional


class EEGSegmentDataset(Dataset):
    """PyTorch Dataset for EEG segments"""
    
    def __init__(self, segments: np.ndarray):
        """
        Args:
            segments: shape (n_segments, n_channels, n_timepoints)
        """
        self.segments = torch.FloatTensor(segments)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        # Return flattened segment for GAN
        segment = self.segments[idx]  # (n_channels, n_timepoints)
        return segment.flatten()  # (n_channels * n_timepoints,)


class Generator(nn.Module):
    """Generator network: noise -> EEG"""
    
    def __init__(self,
                 latent_dim: int = 100,
                 output_dim: Optional[int] = None,
                 hidden_dims: Optional[List[int]] = None):
        super(Generator, self).__init__()
        if output_dim is None:
            output_dim = 64 * 256  # Default: 64 channels x 256 timepoints
        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        layers: List[nn.Module] = []
        in_dim = latent_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = dim
        layers.append(nn.Linear(in_dim, output_dim))
        # Match original notebook behavior: generated EEG is bounded with tanh.
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate EEG from noise"""
        return self.net(z)


class Discriminator(nn.Module):
    """Discriminator network: EEG -> real/fake score"""
    
    def __init__(self,
                 input_dim: Optional[int] = None,
                 hidden_dims: Optional[List[int]] = None):
        super(Discriminator, self).__init__()
        if input_dim is None:
            input_dim = 64 * 256  # Default: 64 channels x 256 timepoints
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim

        layers: List[nn.Module] = []
        in_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            # Match original notebook behavior: LeakyReLU discriminator.
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs fake EEG"""
        return self.net(x)


class TaskGANTrainer:
    """WGAN trainer for task-based EEG"""
    
    def __init__(self, 
                 segments: np.ndarray,
                 task_name: str,
                 device: str = 'cpu',
                 latent_dim: int = 100,
                 batch_size: int = 32,
                 lr: float = 1e-4,
                 n_critic: int = 5,
                 weight_clip: float = 0.01,
                 gen_hidden_dims: Optional[List[int]] = None,
                 disc_hidden_dims: Optional[List[int]] = None,
                 num_workers: int = 0):
        """
        Args:
            segments: preprocessed EEG segments (n_segments, n_channels, n_timepoints)
            task_name: name of the task
            device: 'cpu' or 'cuda'
            latent_dim: dimension of noise vector
            batch_size: training batch size
            lr: learning rate
            n_critic: number of critic updates per generator update (WGAN)
            weight_clip: weight clipping value for WGAN
            gen_hidden_dims: generator hidden layer sizes
            disc_hidden_dims: discriminator hidden layer sizes
            num_workers: DataLoader worker count
        """
        self.task_name = task_name
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr
        self.n_critic = n_critic
        self.weight_clip = weight_clip

        if segments.ndim != 3:
            raise ValueError("segments must have shape (n_segments, n_channels, n_timepoints)")
        if segments.shape[0] == 0:
            raise ValueError("segments is empty; at least one segment is required for training")
        
        # Keep full segment tensor on CPU; batches are moved to device during training.
        self.segments = torch.FloatTensor(segments)
        self.input_dim = segments.shape[1] * segments.shape[2]
        
        # Create data loader
        dataset = EEGSegmentDataset(segments)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda')
        )
        
        # Models
        self.generator = Generator(latent_dim, self.input_dim, gen_hidden_dims).to(self.device)
        self.discriminator = Discriminator(self.input_dim, disc_hidden_dims).to(self.device)
        
        # Optimizers
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Tracking
        self.g_losses = []
        self.d_losses = []
        self.current_epoch = 0
    
    def train_one_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        n_batches = 0
        
        for real_data in self.dataloader:
            real_data = real_data.to(self.device)
            batch_size = real_data.shape[0]
            
            # Train Discriminator
            d_loss_batch = 0.0
            for _ in range(self.n_critic):
                # Zero gradients
                self.opt_d.zero_grad()
                
                # Real data
                d_real = self.discriminator(real_data)
                
                # Fake data
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_data = self.generator(z).detach()
                d_fake = self.discriminator(fake_data)
                
                # WGAN loss (maximized)
                d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
                
                # Backward
                d_loss.backward()
                self.opt_d.step()
                
                # Weight clipping
                for param in self.discriminator.parameters():
                    param.data.clamp_(-self.weight_clip, self.weight_clip)
                
                d_loss_batch += d_loss.item()
            
            d_loss_batch /= self.n_critic
            d_loss_epoch += d_loss_batch
            
            # Train Generator
            self.opt_g.zero_grad()
            
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_data = self.generator(z)
            d_fake = self.discriminator(fake_data)
            
            # Generator loss (minimized)
            g_loss = -torch.mean(d_fake)
            
            # Backward
            g_loss.backward()
            self.opt_g.step()
            
            g_loss_epoch += g_loss.item()
            n_batches += 1
        
        g_loss_epoch /= n_batches
        d_loss_epoch /= n_batches
        
        self.g_losses.append(g_loss_epoch)
        self.d_losses.append(d_loss_epoch)
        self.current_epoch += 1
        
        return g_loss_epoch, d_loss_epoch
    
    def train(self, n_epochs: int = 100, print_freq: int = 10) -> Dict:
        """Train the GAN"""
        print_freq = max(1, print_freq)

        print(f"\nTraining {self.task_name} GAN for {n_epochs} epochs...")
        print(f"  Input dimension: {self.input_dim}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Device: {self.device}")
        
        for epoch in range(n_epochs):
            g_loss, d_loss = self.train_one_epoch()
            
            if (epoch + 1) % print_freq == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs} | G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}")
        
        return {
            'n_epochs': n_epochs,
            'final_g_loss': self.g_losses[-1],
            'final_d_loss': self.d_losses[-1],
            'losses': {
                'g_losses': self.g_losses,
                'd_losses': self.d_losses
            }
        }
    
    def generate_samples(self, n_samples: int = 100) -> np.ndarray:
        """Generate synthetic EEG samples"""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            fake_data = self.generator(z).cpu().numpy()
        
        self.generator.train()
        
        return fake_data
    
    def compute_mmd(self, n_sample_points: int = 256, bandwidth: float = 1.0) -> float:
        """
        Compute Maximum Mean Discrepancy between real and generated data
        """
        self.generator.eval()

        n_samples = min(n_sample_points, len(self.segments))
        if n_samples < 2:
            return float('nan')

        # Sample real data
        real_indices = np.random.choice(len(self.segments), size=n_samples, replace=False)
        real_samples = self.segments[real_indices].view(n_samples, -1).cpu().numpy()

        # Generate fake data matched by sample count
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            fake_samples = self.generator(z).cpu().numpy()

        # Compute MMD using vectorized RBF kernel
        def rbf_kernel_matrix(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
            x2 = np.sum(x * x, axis=1, keepdims=True)
            y2 = np.sum(y * y, axis=1, keepdims=True).T
            dist2 = np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)
            return np.exp(-dist2 / (2.0 * sigma * sigma))

        # K_xx
        k_xx = rbf_kernel_matrix(real_samples, real_samples, bandwidth).mean()

        # K_yy
        k_yy = rbf_kernel_matrix(fake_samples, fake_samples, bandwidth).mean()

        # K_xy
        k_xy = rbf_kernel_matrix(real_samples, fake_samples, bandwidth).mean()

        mmd = np.sqrt(max(0, k_xx + k_yy - 2 * k_xy))

        self.generator.train()

        return mmd
    
    def save(self, output_dir: str):
        """Save trained models and training info"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        torch.save(self.generator.state_dict(), output_dir / f'{self.task_name}_generator.pt')
        torch.save(self.discriminator.state_dict(), output_dir / f'{self.task_name}_discriminator.pt')
        
        # Save training info
        info = {
            'task': self.task_name,
            'latent_dim': self.latent_dim,
            'input_dim': self.input_dim,
            'n_epochs': self.current_epoch,
            'final_g_loss': float(self.g_losses[-1]) if self.g_losses else None,
            'final_d_loss': float(self.d_losses[-1]) if self.d_losses else None,
            'g_losses': [float(x) for x in self.g_losses],
            'd_losses': [float(x) for x in self.d_losses]
        }
        
        with open(output_dir / f'{self.task_name}_training_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Saved models to {output_dir}")
        
        return output_dir


def get_model_dims(model_preset: str) -> Tuple[List[int], List[int]]:
    """Return (generator_hidden_dims, discriminator_hidden_dims) for preset."""
    presets = {
        'small': ([128, 256], [256, 128]),
        # Full preset matches the original resting_state_gans notebook architecture.
        'full': ([256, 512], [512, 256]),
        'base': ([256, 512], [512, 256]),
        'large': ([512, 1024, 2048], [1024, 512, 256]),
    }
    if model_preset not in presets:
        raise ValueError(f"Unknown model_preset '{model_preset}'. Choose from {list(presets.keys())}.")
    return presets[model_preset]


def train_task_gans(preprocessed_data_dir: str, output_dir: str = './task_gan_models',
                   n_epochs: int = 100, device: str = 'cpu',
                   skip_tasks: Optional[List[str]] = None,
                   model_preset: str = 'full',
                   batch_size: int = 32,
                   latent_dim: int = 100,
                   n_critic: int = 5,
                   weight_clip: float = 0.01,
                   max_segments_per_task: Optional[int] = None,
                   num_workers: int = 0) -> Dict:
    """
    Train GANs for all available tasks
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from task_based_eeg_preprocessing import TaskEEGDataManager
    
    if skip_tasks is None:
        skip_tasks = []
    
    print("=" * 80)
    print("Task-Based EEG GAN Training")
    print("=" * 80)
    print(f"Data directory: {preprocessed_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {n_epochs}")
    print(f"Model preset: {model_preset}")
    print(f"Batch size: {batch_size}")

    gen_hidden_dims, disc_hidden_dims = get_model_dims(model_preset)
    
    # Load data manager and find available tasks
    data_manager = TaskEEGDataManager(preprocessed_data_dir)
    summary_df = data_manager.get_task_summary()

    if summary_df.empty:
        print("\nNo preprocessed task data found. Run preprocessing first.")
        return {}
    
    print(f"\nAvailable tasks:")
    print(summary_df)
    
    # Train GANs
    results = {}
    
    for _, row in summary_df.iterrows():
        task_name = row['task']
        
        if task_name in skip_tasks:
            print(f"\nSkipping {task_name}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Processing {task_name}")
        print(f"{'=' * 80}")
        
        try:
            # Load segments
            segments, metadata = data_manager.load_task_segments(task_name)

            if max_segments_per_task is not None and len(segments) > max_segments_per_task:
                keep_idx = np.random.choice(len(segments), size=max_segments_per_task, replace=False)
                segments = segments[keep_idx]
                print(f"Subsampled to {len(segments)} segments (max_segments_per_task={max_segments_per_task})")

            print(f"Loaded {len(segments)} segments: {segments.shape}")
            
            # Create trainer
            trainer = TaskGANTrainer(
                segments=segments,
                task_name=task_name,
                device=device,
                latent_dim=latent_dim,
                batch_size=batch_size,
                n_critic=n_critic,
                weight_clip=weight_clip,
                gen_hidden_dims=gen_hidden_dims,
                disc_hidden_dims=disc_hidden_dims,
                num_workers=num_workers
            )
            
            # Train
            train_result = trainer.train(n_epochs=n_epochs, print_freq=max(1, n_epochs // 5))
            
            # Evaluate
            mmd = trainer.compute_mmd()
            print(f"  MMD (Real vs Generated): {mmd:.4f}")
            
            # Generate samples
            synthetic_samples = trainer.generate_samples(n_samples=100)
            print(f"  Generated {len(synthetic_samples)} synthetic samples: {synthetic_samples.shape}")
            
            # Save
            model_dir = trainer.save(output_dir)
            
            results[task_name] = {
                'train_result': train_result,
                'mmd': float(mmd),
                'n_segments': len(segments),
                'model_dir': str(model_dir)
            }
        
        except Exception as e:
            print(f"Error training {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'=' * 80}")
    print("Training Complete - Summary")
    print(f"{'=' * 80}")
    
    summary_data = []
    for task, res in results.items():
        summary_data.append({
            'task': task,
            'n_segments': res['n_segments'],
            'final_g_loss': res['train_result']['final_g_loss'],
            'final_d_loss': res['train_result']['final_d_loss'],
            'mmd': res['mmd']
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WGAN models for task-based EEG data.')
    parser.add_argument('preprocessed_dir', nargs='?', default='./task_gan_data',
                        help='Directory containing preprocessed task data.')
    parser.add_argument('output_dir', nargs='?', default='./task_gan_models',
                        help='Directory to save trained models.')
    parser.add_argument('n_epochs', nargs='?', type=int, default=100,
                        help='Number of epochs (default: 100).')
    parser.add_argument('--device', default=None, choices=['cpu', 'cuda'],
                        help='Force device. Default auto-detects cuda when available.')
    parser.add_argument('--model-preset', '--model_preset', default='full', choices=['small', 'full', 'base', 'large'],
                        help='Network size preset (default: full).')
    parser.add_argument('--batch-size', '--batch_size', type=int, default=32,
                        help='Batch size (default: 32).')
    parser.add_argument('--latent-dim', '--latent_dim', type=int, default=100,
                        help='Latent noise dimension (default: 100).')
    parser.add_argument('--n-critic', '--n_critic', type=int, default=5,
                        help='Discriminator steps per generator step (default: 5).')
    parser.add_argument('--weight-clip', '--weight_clip', type=float, default=0.01,
                        help='WGAN weight clipping value (default: 0.01).')
    parser.add_argument('--max-segments-per-task', '--max_segments_per_task', type=int, default=None,
                        help='Optional cap to subsample segments per task for memory/runtime control.')
    parser.add_argument('--num-workers', '--num_workers', type=int, default=0,
                        help='DataLoader workers (default: 0).')
    parser.add_argument('--skip-tasks', '--skip_tasks', nargs='*', default=None,
                        help='Optional list of task names to skip.')
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = train_task_gans(
        preprocessed_data_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        device=device,
        skip_tasks=args.skip_tasks,
        model_preset=args.model_preset,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        n_critic=args.n_critic,
        weight_clip=args.weight_clip,
        max_segments_per_task=args.max_segments_per_task,
        num_workers=args.num_workers
    )
