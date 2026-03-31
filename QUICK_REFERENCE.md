# Quick Reference - Task-Based EEG GAN Analysis

## Setup

```bash
# Local machine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Longleaf
module load conda/latest
conda create -n eeg-gans python=3.11 -y
conda activate eeg-gans
pip install -r requirements.txt
```

## Validate Environment

```bash
python test_environment.py
```

## Download Data

```bash
# Full dataset (172 GB, ~2-4 hours)
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 --no-sign-request

# Test set (quick, ~100 MB)
aws s3 sync s3://openneuro.org/ds005515/sub-NDARAA/eeg ./raw/ds005515/ --no-sign-request

# Resume interrupted download (safe to re-run)
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 --no-sign-request
```

## Preprocessing

### Quick Test (single file)

```bash
python << 'EOF'
from task_based_eeg_preprocessing import HBNTaskDataLoader, TaskEEGPreprocessor
loader = HBNTaskDataLoader('./raw/ds005515')
raw_list = loader.load_task_data('DiaryOfAWimpyKid', max_files=1)
preprocessor = TaskEEGPreprocessor()
segments, metadata = preprocessor.process_task_files(raw_list)
print(f"Processed {len(segments)} segments: {segments.shape}")
EOF
```

### Full Pipeline (all tasks)

```bash
python task_based_eeg_preprocessing.py ./raw/ds005515 ./task_gan_data
```

### With Custom Parameters

```bash
python << 'EOF'
from task_based_eeg_preprocessing import process_all_hbn_tasks
process_all_hbn_tasks(
    raw_data_dir='./raw/ds005515',
    output_dir='./task_gan_data',
    segment_duration=2.0,        # seconds
    max_files_per_task=None      # None = all files
)
EOF
```

## Check Preprocessed Data

```bash
python << 'EOF'
from task_based_eeg_preprocessing import TaskEEGDataManager
manager = TaskEEGDataManager('./task_gan_data')
print(manager.get_task_summary())
EOF
```

## GAN Training

### Quick Test (CPU)

```bash
python << 'EOF'
from task_based_eeg_gan import train_task_gans
train_task_gans('./task_gan_data', './task_gan_models', n_epochs=5, device='cpu')
EOF
```

### Full Training (GPU recommended)

```bash
python task_based_eeg_gan.py ./task_gan_data ./task_gan_models 100
```

### Command-line with Options

```bash
python << 'EOF'
import torch
from task_based_eeg_gan import train_task_gans
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_task_gans('./task_gan_data', './task_gan_models', n_epochs=100, device=device)
EOF
```

## Analysis & Visualization

### Load and Check Results

```bash
python << 'EOF'
import pandas as pd
import json
from pathlib import Path

results = []
for task_dir in Path('./task_gan_models').iterdir():
    meta_file = task_dir / f'{task_dir.name}_training_info.json'
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        results.append({
            'task': meta['task'],
            'epochs': meta['n_epochs'],
            'g_loss': meta['final_g_loss'],
            'd_loss': meta['final_d_loss']
        })

df = pd.DataFrame(results)
print(df)
print(f"\nModels saved to: ./task_gan_models/")
print(f"Models: {len(results)} tasks trained")
EOF
```

### Jupyter Notebook

```bash
jupyter notebook task_based_eeg_analysis.ipynb
```

### Generate Synthetic EEG

```bash
python << 'EOF'
import torch
from task_based_eeg_preprocessing import TaskEEGDataManager
from task_based_eeg_gan import TaskGANTrainer

# Load real data
manager = TaskEEGDataManager('./task_gan_data')
segments, metadata = manager.load_task_segments('DiaryOfAWimpyKid')

# Create trainer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = TaskGANTrainer(segments, 'DiaryOfAWimpyKid', device=device)

# Load trained generator
trainer.generator.load_state_dict(
    torch.load('./task_gan_models/DiaryOfAWimpyKid_generator.pt', 
               map_location=device)
)

# Generate samples
synthetic = trainer.generate_samples(n_samples=100)
print(f"Generated {len(synthetic)} samples: {synthetic.shape}")

# Optional: save synthetic samples
import numpy as np
np.save('synthetic_DiaryOfAWimpyKid.npy', synthetic)
EOF
```

## Longleaf HPC

### Initial Setup

```bash
ssh <onyen>@longleaf.unc.edu
cd /work/users/$USER
mkdir eeg-project && cd eeg-project

# Copy files
scp <local>:~/amplab-digtwins/*.py .
scp <local>:~/amplab-digtwins/*.md .
scp <local>:~/amplab-digtwins/requirements.txt .

# Environment
module load conda/latest
conda create -n eeg-gans python=3.11 -y
conda activate eeg-gans
pip install -r requirements.txt
```

### Download Data on Longleaf

```bash
module load aws-cli/2.13.30
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 --no-sign-request
```

### Submit Jobs

```bash
# Interactive testing (GPU)
srun --pty -p gpu -t 01:00:00 --gpus=1 -N 1 --mem=32g bash
module load conda/latest
conda activate eeg-gans
python test_environment.py

# Preprocessing (CPU job)
sbatch << 'EOF'
#!/bin/bash
#SBATCH -p general -t 08:00:00 --mem=64g --cpus-per-task=8
module load conda/latest
conda activate eeg-gans
python task_based_eeg_preprocessing.py ./raw/ds005515 ./task_gan_data
EOF

# GAN Training (GPU job)
sbatch << 'EOF'
#!/bin/bash
#SBATCH -p gpu -t 12:00:00 --gpus=1 --mem=64g --cpus-per-task=8
module load conda/latest
conda activate eeg-gans
python task_based_eeg_gan.py ./task_gan_data ./task_gan_models 100
EOF
```

### Monitor Jobs

```bash
squeue -u $USER
squeue -u $USER -j <job_id>
sacct -u $USER --format=JobID,State,Elapsed
tail -f slurm-<job_id>.out
```

### Transfer Results Back

```bash
# From local machine
scp -r <onyen>@longleaf.unc.edu:/work/users/<onyen>/eeg-project/task_gan_models ./

# Or just models (smaller)
scp <onyen>@longleaf.unc.edu:/work/users/<onyen>/eeg-project/task_gan_models/*.pt ./
scp <onyen>@longleaf.unc.edu:/work/users/<onyen>/eeg-project/task_gan_models/*.json ./
```

## Troubleshooting

### Test if everything works

```bash
python test_environment.py
```

### Show available tasks

```bash
python << 'EOF'
from task_based_eeg_preprocessing import HBNTaskDataLoader
loader = HBNTaskDataLoader('./raw/ds005515')
print(loader.get_task_metadata())
EOF
```

### Check GPU availability

```bash
python << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF
```

### List all preprocessed tasks

```bash
ls -la task_gan_data/
```

### List all trained models

```bash
ls -la task_gan_models/*.pt
```

### Show preprocessing details

```bash
python << 'EOF'
import json
from pathlib import Path

for meta_file in Path('./task_gan_data').glob('*/\*_metadata.json'):
    with open(meta_file) as f:
        meta = json.load(f)
    print(f"{meta_file.parent.name}:")
    print(f"  Segments: {meta['n_segments']}")
    print(f"  Channels: {meta['n_channels']}")
    print(f"  Duration: {meta['n_timepoints']/meta['sfreq']:.1f}s @ {meta['sfreq']}Hz")
EOF
```

## Performance Estimation

### Time

- **Preprocessing**: 30 min (125 files, 8 CPUs) or 10 min (8 GPUs)
- **GAN training**: 2 hrs/task (CPU) or 15 min/task (GPU)
- **Full pipeline**: ~14 hrs (CPU) or 2 hrs (GPU)

### Storage

- **Raw data**: 172 GB
- **Preprocessed**: 50-80 GB
- **Models**: 200 MB
- **Total**: ~220-250 GB

## Files Generated

After running full pipeline:

```
raw/ds005515/                              # Original .set files (172 GB)
task_gan_data/
├── DiaryOfAWimpyKid/
│   ├── DiaryOfAWimpyKid_segments.npy
│   ├── DiaryOfAWimpyKid_metadata.json
│   └── DiaryOfAWimpyKid_segments_individual/      # Individual segment files
├── ThePresent/
├── FunwithFractals/
├── DespicableMe/
├── contrastChangeDetection/
├── surroundSupp/
└── symbolSearch/

task_gan_models/
├── DiaryOfAWimpyKid_generator.pt
├── DiaryOfAWimpyKid_discriminator.pt
├── DiaryOfAWimpyKid_training_info.json
├── ThePresent_generator.pt
├── ThePresent_discriminator.pt
├── ThePresent_training_info.json
├── ... (one set per task)
```

## Common Tasks

| Task | Command |
|------|---------|
| Validate setup | `python test_environment.py` |
| Quick test | `python << 'EOF' ... EOF` (see examples above) |
| Full preprocessing | `python task_based_eeg_preprocessing.py ./raw/ds005515 ./task_gan_data` |
| Full GAN training | `python task_based_eeg_gan.py ./task_gan_data ./task_gan_models 100` |
| Jupyter analysis | `jupyter notebook task_based_eeg_analysis.ipynb` |
| Check results | `python << 'EOF' ... EOF` (see examples above) |
| Download data | `aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 --no-sign-request` |

## More Info

- **Full docs**: See [README_TASK_BASED_EEG.md](README_TASK_BASED_EEG.md)
- **Longleaf setup**: See [LONGLEAF_SETUP.md](LONGLEAF_SETUP.md)
- **Jupyter notebook**: [task_based_eeg_analysis.ipynb](task_based_eeg_analysis.ipynb)
- **Module docs**: Check docstrings in .py files
