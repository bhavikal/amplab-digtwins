# Task-Based EEG GAN Analysis

Preprocessing, analysis, and synthetic generation of task-based EEG recordings from the Healthy Brain Network (HBN) Release 10 dataset using Wasserstein GANs.

## Overview

This repository contains code to:

1. **Preprocess** task-based EEG recordings from the HBN dataset (ds005515)
2. **Extract** fixed-duration segments from task events
3. **Train** task-specific Wasserstein GANs (WGANs) for synthetic EEG generation
4. **Evaluate** generated vs real EEG using spectral and statistical measures
5. **Compare** task characteristics across cognitive tasks

### Supported Tasks

- DiaryOfAWimpyKid (26 files)
- ThePresent (19 files)
- FunwithFractals (19 files)
- DespicableMe (15 files)
- contrastChangeDetection (14 files)
- surroundSupp (8 files)
- symbolSearch (2 files)

## Installation

### Local Setup

```bash
# Clone or download this repository
git clone <repository-url> amplab-digtwins
cd amplab-digtwins

# Create Python environment (Python 3.9+)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_environment.py
```

### On Longleaf HPC

See [LONGLEAF_SETUP.md](LONGLEAF_SETUP.md) for detailed instructions on setting up this analysis on the Longleaf cluster at UNC.

```bash
# Quick start on Longleaf
module load conda/latest
conda create -n eeg-gans python=3.11 -y
conda activate eeg-gans
pip install -r requirements.txt
sbatch longleaf_preprocess.sh  # Example SLURM script
```

## Quick Start

### 1. Download Data

The HBN EEG Release 10 dataset (172 GB) is available on OpenNeuro via AWS S3:

```bash
# Install AWS CLI (if not already installed)
pip install awscli-local  # or: brew install awscli (on macOS)

# Download data (first 10 files for testing)
aws s3 sync s3://openneuro.org/ds005515/sub-NDARAA/eeg ./raw/ds005515/ --no-sign-request --exclude "*" --include "sub-NDARAA*"

# Or download all data (requires ~172 GB storage and several hours)
module load aws-cli  # On Longleaf
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515/ --no-sign-request
```

### 2. Preprocess Task-Based EEG

```bash
# Define paths
export RAW_DATA="./raw/ds005515"
export OUTPUT_DATA="./task_gan_data"

# Run preprocessing
python -c "
from task_based_eeg_preprocessing import process_all_hbn_tasks
process_all_hbn_tasks(
    raw_data_dir='$RAW_DATA',
    output_dir='$OUTPUT_DATA',
    segment_duration=2.0
)
"

# Or use the CLI
python task_based_eeg_preprocessing.py $RAW_DATA $OUTPUT_DATA
```

**What it does:**
- Loads .set files for each task
- Applies bandpass filter (0.5-45 Hz)
- Downsamples to 128 Hz
- Re-references to average
- Extracts 2-second non-overlapping segments
- Per-segment Z-score normalization
- Saves as: segments.npy, metadata.json, individual .npy files

### 3. Train Task-Based GANs

```bash
# CPU-based (slower, no GPU needed)
python -c "
from task_based_eeg_gan import train_task_gans
train_task_gans(
    preprocessed_data_dir='./task_gan_data',
    output_dir='./task_gan_models',
    n_epochs=100,
    device='cpu'
)
"

# GPU-based (recommended, 5-10x faster)
python task_based_eeg_gan.py ./task_gan_data ./task_gan_models 100
```

**What it does:**
- Loads preprocessed segments for each task
- Creates discriminator and generator networks
- Implements Wasserstein GAN training with weight clipping
- Computes Maximum Mean Discrepancy (MMD) metric
- Saves trained models (.pt files) and training metrics

### 4. Analyze Results

```bash
# Run interactive Jupyter notebook
jupyter notebook task_based_eeg_analysis.ipynb
```

Or run Python script for batch processing:

```bash
python << 'EOF'
from task_based_eeg_preprocessing import TaskEEGDataManager
from task_based_eeg_gan import TaskGANTrainer
import torch

# Load results
data_manager = TaskEEGDataManager('./task_gan_data')
summary = data_manager.get_task_summary()
print(summary)

# Load and evaluate a task
segments, metadata = data_manager.load_task_segments('DiaryOfAWimpyKid')
trainer = TaskGANTrainer(segments, 'DiaryOfAWimpyKid', device='cuda')
trainer.generator.load_state_dict(torch.load('./task_gan_models/DiaryOfAWimpyKid_generator.pt'))

# Generate synthetic samples
synthetic = trainer.generate_samples(n_samples=100)
print(f"Generated {len(synthetic)} synthetic samples")
EOF
```

## File Structure

```
amplab-digtwins/
├── task_based_eeg_preprocessing.py      # Preprocessing module
├── task_based_eeg_gan.py                # GAN training module
├── task_based_eeg_analysis.ipynb        # Full workflow notebook
├── test_environment.py                  # Environment validation script
├── requirements.txt                     # Python dependencies
├── LONGLEAF_SETUP.md                   # Longleaf HPC setup guide
└── README.md                            # This file

# After running:
├── raw/ds005515/                        # Downloaded EEG .set files
│   ├── sub-NDARAA/eeg/
│   ├── sub-NDAABB/eeg/
│   └── ...
├── task_gan_data/                       # Preprocessed segments
│   ├── DiaryOfAWimpyKid/
│   │   ├── DiaryOfAWimpyKid_segments.npy
│   │   ├── DiaryOfAWimpyKid_metadata.json
│   │   └── DiaryOfAWimpyKid_segments_individual/
│   ├── ThePresent/
│   └── ...
└── task_gan_models/                     # Trained models
    ├── DiaryOfAWimpyKid_generator.pt
    ├── DiaryOfAWimpyKid_discriminator.pt
    ├── DiaryOfAWimpyKid_training_info.json
    └── ...
```

## Module Documentation

### task_based_eeg_preprocessing.py

**HBNTaskDataLoader**
- Organizes .set files by task type
- Provides interface to load specific tasks

**TaskEEGPreprocessor**
- Applies standard EEG preprocessing pipeline
- Filters, resamples, re-references data
- Extracts and normalizes segments

**TaskEEGDataManager**
- Saves/loads preprocessed segments
- Organizes outputs by task
- Provides metadata management

**process_all_hbn_tasks()**
- Convenience function to process all tasks at once

### task_based_eeg_gan.py

**TaskGANTrainer**
- Wasserstein GAN implementation
- WGAN loss with weight clipping
- MMD metric computation
- Model saving/loading

**EEGSegmentDataset**
- PyTorch Dataset wrapper for segments

**train_task_gans()**
- Trains GANs for all available tasks
- Returns training metrics and MMD scores

## Performance Notes

**Typical runtimes:**

| Operation | CPU | GPU |
|-----------|-----|-----|
| Preprocess (~125 files) | 30 min | 10 min* |
| Train 1 task (100 epochs) | 2 hrs | 15 min |
| Full pipeline (7 tasks) | ~14 hrs | 2 hrs |

*Preprocessing is CPU-bound; GPU speedup depends on I/O

**Storage requirements:**

| Component | Size |
|-----------|------|
| Raw .set files (full) | 172 GB |
| Preprocessed segments | 50-80 GB |
| Trained models (all tasks) | 200 MB |
| Visualizations | 1-2 GB |

## Example: Comparing Real vs Synthetic EEG

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Load data
from task_based_eeg_preprocessing import TaskEEGDataManager
from task_based_eeg_gan import TaskGANTrainer

data_manager = TaskEEGDataManager('./task_gan_data')
real_segments, metadata = data_manager.load_task_segments('ThePresent')

# Initialize trainer and load model
trainer = TaskGANTrainer(real_segments, 'ThePresent')
trainer.generator.load_state_dict(
    torch.load('./task_gan_models/ThePresent_generator.pt')
)

# Generate synthetic
synthetic = trainer.generate_samples(n_samples=100)

# Compare power spectra
sfreq = metadata['sfreq']
freqs_real, psd_real = welch(real_segments[0], sfreq)
freqs_synth, psd_synth = welch(synthetic[0], sfreq)

plt.semilogy(freqs_real, psd_real, label='Real')
plt.semilogy(freqs_synth, psd_synth, label='Synthetic', linestyle='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.legend()
plt.show()
```

## Troubleshooting

### Memory Issues

**Problem:** Out of memory errors during preprocessing
**Solutions:**
- Reduce segment duration: `segment_duration=1.0`
- Limit files per task: `max_files_per_task=10`
- Use external SSD for raw data
- On Longleaf: Use GPU node with more memory

### GPU Not Detected

**Problem:** CUDA not available despite having GPU
**Solutions:**
```bash
# Check GPU visibility
nvidia-smi
nvcc --version

# Verify PyTorch can access GPU
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU if GPU issues persist
python task_based_eeg_gan.py ./task_gan_data ./task_gan_models 100 --device cpu
```

### Data Download Issues

**Problem:** S3 sync fails or is slow
**Solutions:**
```bash
# Resume interrupted sync (idempotent)
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 --no-sign-request

# Increase concurrency
aws s3 sync ... --max-concurrent-requests 20

# Download specific tasks only
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 \
    --no-sign-request --exclude "*" --include "*DespicableMe*"
```

### Missing Dependencies

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Check installation
pip list | grep -E "torch|mne|scipy"
```

## Testing

Validate environment before running full pipeline:

```bash
python test_environment.py
```

This checks:
- ✓ Required packages installed
- ✓ Data files accessible
- ✓ Preprocessing pipeline works
- ✓ GAN training works on synthetic data

## Citation

If you use this code, please cite:

```bibtex
@software{amplab_task_eeg_gan,
  title={Task-Based EEG GAN Analysis for HBN Dataset},
  author={...},
  year={2024},
  url={https://github.com/...}
}
```

## Dataset Citation

**HBN EEG Release 10 (ds005515):**

Alexander, L. M., Escalera, J., Ai, L., et al. (2017). 
"An open resource for transdiagnostic research in pediatric mental health and learning disorders." 
Scientific Data, 4, 170181.

Available on OpenNeuro: https://openneuro.org/datasets/ds005515

## License

This code is provided as-is for research purposes.

Dataset license: CC-BY-SA 4.0

## Related Resources

- **MNE-Python**: https://mne.tools/ - EEG processing library
- **PyTorch**: https://pytorch.org/ - Deep learning framework
- **Longleaf HPC**: https://its.unc.edu/longleaf/ - UNC computing cluster
- **BIDS Standard**: https://bids-standard.github.io/ - Brain data format
- **OpenNeuro**: https://openneuro.org/ - Open neuroimaging data

## Authors

- Bhavika Lingutla

## Changelog

### Version 1.0 (2024-XX-XX)
- Initial release
- Task-based preprocessing pipeline
- WGAN training implementation
- Longleaf HPC support
- Full Jupyter workflow notebook

---

For questions or issues, please open an issue on GitHub or contact the project maintainers.
