"""
Quick Test Script for Task-Based EEG Analysis

This script validates the environment and runs a minimal preprocessing/GAN training example.
Useful for quick validation before running full pipeline.
"""

import sys
import os

def test_imports():
    """Test critical imports"""
    print("Testing imports...")
    try:
        import mne
        import torch
        import numpy as np
        import pandas as pd
        from scipy import signal
        import matplotlib.pyplot as plt
        print("✓ All imports successful")
        print(f"  - MNE version: {mne.__version__}")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - NumPy version: {np.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    - GPU: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_data_location(raw_data_dir='./raw/ds005515'):
    """Test data location"""
    print(f"\nTesting data location: {raw_data_dir}")
    
    if not os.path.exists(raw_data_dir):
        print(f"✗ Data directory not found: {raw_data_dir}")
        print("  Download with: aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 --no-sign-request")
        return False
    
    # Count .set files
    import glob
    set_files = glob.glob(os.path.join(raw_data_dir, 'sub-*/eeg/*_eeg.set'))
    
    if len(set_files) == 0:
        print(f"✗ No .set files found in {raw_data_dir}")
        return False
    
    print(f"✓ Data directory found with {len(set_files)} .set files")
    
    # Show sample tasks
    tasks = {}
    for fpath in set_files[:10]:
        basename = os.path.basename(fpath)
        if 'task-' in basename:
            task = basename.split('task-')[1].split('_')[0]
            tasks[task] = tasks.get(task, 0) + 1
    
    if tasks:
        print(f"  Sample tasks found:")
        for task, count in tasks.items():
            print(f"    - {task}: {count} files")
    
    return True


def test_preprocessing(raw_data_dir='./raw/ds005515', output_dir='./test_preprocess_output'):
    """Test preprocessing on a single file"""
    print(f"\nTesting preprocessing...")
    
    if not os.path.exists(raw_data_dir):
        print("✗ Skipping preprocessing test (data not found)")
        return False
    
    try:
        import glob
        import mne
        from task_based_eeg_preprocessing import TaskEEGPreprocessor
        
        # Find first .set file
        set_files = glob.glob(os.path.join(raw_data_dir, 'sub-*/eeg/*_eeg.set'))
        if not set_files:
            print("✗ No .set files found")
            return False
        
        test_file = set_files[0]
        print(f"  Loading: {os.path.basename(test_file)}")
        
        # Load data
        raw = mne.io.read_raw_eeglab(test_file, preload=True)
        print(f"  ✓ Loaded: {raw.info['sfreq']} Hz, {len(raw.ch_names)} channels, {raw.n_times} samples")
        
        # Preprocess
        preprocessor = TaskEEGPreprocessor()
        raw_proc = preprocessor.preprocess_raw(raw.copy())
        print(f"  ✓ Preprocessed")
        
        # Extract segments
        segments = preprocessor.extract_segments(raw_proc, segment_duration=1.0)
        print(f"  ✓ Extracted segments: {segments.shape}")
        
        # Normalize
        normalized = preprocessor.normalize_segments(segments)
        print(f"  ✓ Normalized segments: {normalized.shape}")
        
        print("✓ Preprocessing test passed")
        return True
    
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gan_training(n_test_samples=50, n_epochs=5):
    """Test GAN training on synthetic data"""
    print(f"\nTesting GAN training...")
    
    try:
        import torch
        import numpy as np
        from task_based_eeg_gan import TaskGANTrainer
        
        # Create synthetic data
        n_channels = 64
        n_timepoints = 256
        synthetic_segments = np.random.randn(n_test_samples, n_channels, n_timepoints).astype(np.float32)
        
        print(f"  Created synthetic test data: {synthetic_segments.shape}")
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Using device: {device}")
        
        # Train
        trainer = TaskGANTrainer(
            segments=synthetic_segments,
            task_name='test_task',
            device=device,
            batch_size=8
        )
        
        print(f"  ✓ Created GAN trainer")
        
        # Quick training (just a few steps to verify)
        for epoch in range(n_epochs):
            g_loss, d_loss = trainer.train_one_epoch()
            print(f"    Epoch {epoch+1}/{n_epochs}: G Loss {g_loss:.4f}, D Loss {d_loss:.4f}")
        
        # Generate
        synthetic = trainer.generate_samples(n_samples=10)
        print(f"  ✓ Generated samples: {synthetic.shape}")
        
        # MMD
        mmd = trainer.compute_mmd()
        print(f"  ✓ Computed MMD: {mmd:.4f}")
        
        print("✓ GAN training test passed")
        return True
    
    except Exception as e:
        print(f"✗ GAN training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_test():
    """Run all tests"""
    print("=" * 80)
    print("Task-Based EEG Analysis - Environment Validation")
    print("=" * 80)
    
    results = {
        'imports': test_imports(),
        'data_location': test_data_location(),
        'preprocessing': test_preprocessing(),
        'gan_training': test_gan_training()
    }
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! Environment is ready.")
    else:
        print("\n✗ Some tests failed. Check output above for details.")
        print("\nCommon issues:")
        print("  - Missing data: Download with aws s3 sync command")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - GPU issues: Check nvidia-smi or use device='cpu'")
    
    return all_passed


if __name__ == '__main__':
    success = run_full_test()
    sys.exit(0 if success else 1)
