# Task-Based EEG Analysis on Longleaf HPC

This guide explains how to transfer and run the task-based EEG GAN analysis on the Longleaf HPC cluster at UNC.

## Quick Start

```bash
# On Longleaf
cd ~/projects/eeg-gans  # or your project directory

# 1. Copy Python modules
scp <your_machine>:~/amplab-digtwins/task_based_eeg_*.py .
scp <your_machine>:~/amplab-digtwins/requirements.txt .
scp -r <your_machine>:~/amplab-digtwins/slurm .

# 2. Set up Python environment using conda
module load conda/latest
conda create -n eeg-gans python=3.11 -y
conda activate eeg-gans
pip install -r requirements.txt

# 3. Download full HBN dataset from OpenNeuro AWS
module load aws-cli/2.13.30
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 --no-sign-request
# This downloads ~172GB and may take several hours

# 4. Run preprocessing interactively or submit SLURM job (see below)
# Full pipeline with defaults
sbatch slurm/full_pipeline.sbatch

# Smoke test (small model + limited data)
sbatch --export=ALL,EPOCHS=10,MODEL_PRESET=small,BATCH=8,MAX_FILES=3,MAX_SEGS=1000 slurm/full_pipeline.sbatch
```

## Detailed Setup

### 1. Access Longleaf

```bash
ssh <onyen>@longleaf.unc.edu
```

### 2. Create Project Directory

```bash
cd ~/projects  # or /work/users/<onyen>/ for larger storage
mkdir eeg-gans
cd eeg-gans
```

### 3. Transfer Code

From your local machine:

```bash
# Transfer Python modules
scp task_based_eeg_preprocessing.py <onyen>@longleaf.unc.edu:~/projects/eeg-gans/
scp task_based_eeg_gan.py <onyen>@longleaf.unc.edu:~/projects/eeg-gans/
scp task_based_eeg_analysis.ipynb <onyen>@longleaf.unc.edu:~/projects/eeg-gans/
scp requirements.txt <onyen>@longleaf.unc.edu:~/projects/eeg-gans/
```

Or use git if the code is in a repository:

```bash
# On Longleaf
cd ~/projects/eeg-gans
git clone <your-repo-url> .
```

### 4. Set Up Python Environment

```bash
# Load conda module
module load conda/latest

# Create environment
conda create -n eeg-gans python=3.11 numpy scipy pandas matplotlib -y

# Activate and install
conda activate eeg-gans
pip install -r requirements.txt

# Verify installation
python -c "import torch, mne; print(f'PyTorch: {torch.__version__}'); print(f'MNE: {mne.__version__}')"
```

### 5. Download Data from OpenNeuro

```bash
# Load AWS CLI
module load aws-cli/2.13.30

# Create data directory
mkdir -p raw/ds005515

# Sync full dataset (~172GB)
# Use the --help option to see resume capability
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 --no-sign-request

# Alternative: Sync in background with nohup
nohup aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 --no-sign-request > s3_sync.log 2>&1 &

# Monitor progress
tail -f s3_sync.log
```

## Running Analysis

### Option 1: Interactive (for testing)

```bash
# Request interactive resource
srun --pty -p gpu -t 01:00:00 -N 1 --mem=32g --gpus=1 bash

# Load modules
module load conda/latest
conda activate eeg-gans

# Test with subset of data (quick validation)
python task_based_eeg_preprocessing.py ./raw/ds005515 ./task_gan_data_test 2>&1 | head -100

# Or use Python directly
python << 'EOF'
from task_based_eeg_preprocessing import process_all_hbn_tasks
results = process_all_hbn_tasks(
    raw_data_dir='./raw/ds005515',
    output_dir='./task_gan_data',
    segment_duration=2.0,
    max_files_per_task=3  # Limit for testing
)
print(results)
EOF
```

### Option 2: Batch SLURM Jobs

#### Preprocessing Job

Create `preprocess_tasks.sh`:

```bash
#!/bin/bash
#SBATCH -J hbn-preprocess
#SBATCH -p general
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH -o preprocess_%j.out
#SBATCH -e preprocess_%j.err

module load conda/latest
conda activate eeg-gans

cd $SLURM_SUBMIT_DIR

python task_based_eeg_preprocessing.py \
    ./raw/ds005515 \
    ./task_gan_data \
    --segment_duration 2.0

echo "Preprocessing complete!"
```

Submit:

```bash
sbatch preprocess_tasks.sh
```

#### GAN Training Job (GPU)

Create `train_gans.sh`:

```bash
#!/bin/bash
#SBATCH -J hbn-gan-train
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus=1
#SBATCH -d singleton
#SBATCH -o gan_train_%j.out
#SBATCH -e gan_train_%j.err

module load conda/latest
conda activate eeg-gans

cd $SLURM_SUBMIT_DIR

python << 'EOF'
import torch
from task_based_eeg_gan import train_task_gans

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

results = train_task_gans(
    preprocessed_data_dir='./task_gan_data',
    output_dir='./task_gan_models',
    n_epochs=100,
    device=device
)

print("GAN training complete!")
EOF
```

Submit:

```bash
sbatch train_gans.sh
```

#### Full Pipeline Script

Create `run_full_pipeline.sh`:

```bash
#!/bin/bash
#SBATCH -J hbn-full-pipeline
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus=1
#SBATCH -o full_pipeline_%j.out
#SBATCH -e full_pipeline_%j.err

module load conda/latest
conda activate eeg-gans

cd $SLURM_SUBMIT_DIR

echo "=== Starting HBN Task-Based EEG Analysis Pipeline ==="
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi -L)"

# Step 1: Preprocess
echo ""
echo "Step 1: Preprocessing task-based EEG..."
python -c "
from task_based_eeg_preprocessing import process_all_hbn_tasks
process_all_hbn_tasks(
    raw_data_dir='./raw/ds005515',
    output_dir='./task_gan_data',
    segment_duration=2.0
)
"

if [ $? -ne 0 ]; then
    echo "Preprocessing failed!"
    exit 1
fi

# Step 2: Train GANs
echo ""
echo "Step 2: Training task-based GANs..."
python -c "
import torch
from task_based_eeg_gan import train_task_gans

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_task_gans(
    preprocessed_data_dir='./task_gan_data',
    output_dir='./task_gan_models',
    n_epochs=100,
    device=device
)
"

if [ $? -ne 0 ]; then
    echo "GAN training failed!"
    exit 1
fi

echo ""
echo "=== Pipeline Complete ==="
echo "Time: $(date)"
echo "Results saved to:"
echo "  - Preprocessed data: ./task_gan_data/"
echo "  - Trained models: ./task_gan_models/"
```

Submit:

```bash
sbatch run_full_pipeline.sh
```

### Monitor Jobs

```bash
# View all your jobs
squeue -u $USER

# View specific job details
squeue -u $USER -j <job_id>

# Cancel a job
scancel <job_id>

# View completed job info
sacct -u $USER -j <job_id>
```

## Data Management

### Downloading Dataset

The full HBN EEG Release 10 dataset is 172.4 GB. Longleaf users have access to `/work/` directories with large quotas.

```bash
# Check quota
quota

# Use /work for large data
cd /work/users/$USER
mkdir -p eeg-project
cd eeg-project

# Download specific task
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg/ ./raw/ds005515/ \
    --no-sign-request \
    --exclude "*" \
    --include "*DespicableMe*"  # Example: just one task

# Or all tasks (full download)
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg/ ./raw/ds005515/ \
    --no-sign-request
```

### Storage Estimates

- Raw .set files: ~172 GB
- Preprocessed segments (.npy): ~50-80 GB (depends on segment length)
- Trained models (.pt files): ~100-200 MB per task
- Visualizations: ~1-2 GB

Consider using `/work/` which typically has:
- Higher storage quotas (100s of GB to 1-2 TB)
- Better for I/O intensive operations
- Regular backups not enabled (suitable for reproducible computations)

## Analyzing Results

### Transfer Results Back

```bash
# From local machine
scp -r <onyen>@longleaf.unc.edu:~/projects/eeg-gans/task_gan_models ./
scp -r <onyen>@longleaf.unc.edu:~/projects/eeg-gans/task_gan_data ./

# Or just model files (smaller)
scp <onyen>@longleaf.unc.edu:~/projects/eeg-gans/task_gan_models/*.pt ./
scp <onyen>@longleaf.unc.edu:~/projects/eeg-gans/task_gan_models/*.json ./
```

### Using Jupyter on Longleaf

```bash
# Start Jupyter notebook on compute node
srun --pty -p gpu -t 02:00:00 -N 1 --mem=32g --gpus=1 bash

# In compute node:
module load conda/latest
conda activate eeg-gans
jupyter notebook --ip=0.0.0.0 --no-browser --port=8888

# On local machine (in another terminal):
ssh -N -L 8888:longleaf-gpu-node01:8888 <onyen>@longleaf.unc.edu

# Then open browser to: http://localhost:8888
```

## Troubleshooting

### Issue: Module not found errors

```bash
# Make sure Python environment is activated
conda activate eeg-gans

# Verify packages are installed
pip list | grep torch
pip list | grep mne

# Reinstall if needed
pip install --upgrade torch mne
```

### Issue: GPU not available

```bash
# Check GPU allocation in SLURM
#SBATCH --gpus=1  # Add to script

# Verify GPU is visible
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"

# If False, check nvidia-smi
nvidia-smi
```

### Issue: Out of memory errors

Reduce batch size or segment length:

```python
# In preprocessing
segment_duration=1.0  # Instead of 2.0

# In GAN training
batch_size=16  # Instead of 32
```

### Issue: AWS S3 sync is slow

Use parallel downloads:

```bash
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 \
    --no-sign-request \
    --max-concurrent-requests 10
```

## Output Structure

```
projects/eeg-gans/
├── raw/ds005515/                              # Downloaded data
│   ├── sub-NDARAA/eeg/
│   ├── sub-NDAABB/eeg/
│   └── ...
├── task_gan_data/                            # Preprocessed segments
│   ├── DiaryOfAWimpyKid/
│   │   ├── DiaryOfAWimpyKid_segments.npy
│   │   ├── DiaryOfAWimpyKid_metadata.json
│   │   └── DiaryOfAWimpyKid_segments_individual/
│   ├── ThePresent/
│   └── ...
├── task_gan_models/                          # Trained models
│   ├── DiaryOfAWimpyKid_generator.pt
│   ├── DiaryOfAWimpyKid_discriminator.pt
│   ├── DiaryOfAWimpyKid_training_info.json
│   └── ...
├── task_based_eeg_preprocessing.py
├── task_based_eeg_gan.py
├── task_based_eeg_analysis.ipynb
└── requirements.txt
```

## Performance Notes

On Longleaf with typical settings:

- **Preprocessing**: ~30 minutes for 125 files on 8 CPUs
- **GAN Training (CPU)**: ~2 hours per task for 100 epochs
- **GAN Training (GPU)**: ~15 minutes per task for 100 epochs
- **Full pipeline**: ~3 hours with GPU (preprocessing + all 7 tasks)

## Example: Complete Workflow

```bash
# Login and setup
ssh <onyen>@longleaf.unc.edu
cd /work/users/$USER
mkdir eeg-project && cd eeg-project

# Get code
git clone https://github.com/<your-username>/amplab-digtwins.git .
# Or upload files via scp

# Setup
module load conda/latest
conda create -n eeg-gans python=3.11 -y
conda activate eeg-gans
pip install -r requirements.txt

# Download data (~2 hours for 172GB)
module load aws-cli/2.13.30
aws s3 sync s3://openneuro.org/ds005515/sub-*/eeg ./raw/ds005515 --no-sign-request

# Run with GPU
sbatch -p gpu -t 12:00:00 --gpus=1 -N 1 --mem=64g << 'EOF'
#!/bin/bash
module load conda/latest
conda activate eeg-gans
cd $SLURM_SUBMIT_DIR
python -c "
from task_based_eeg_preprocessing import process_all_hbn_tasks
from task_based_eeg_gan import train_task_gans
import torch

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
process_all_hbn_tasks('./raw/ds005515', './task_gan_data')
train_task_gans('./task_gan_data', './task_gan_models', n_epochs=100, device=dev)
"
EOF

# Monitor
squeue -u $USER

# Download results
scp -r <onyen>@longleaf.unc.edu:/work/users/<onyen>/eeg-project/task_gan_models ./
```

## Support

For Longleaf-specific questions:
- Documentation: https://its.unc.edu/longleaf/
- Email: ITS Help Desk (support@unc.edu)

For EEG/GAN questions:
- MNE documentation: https://mne.tools/
- PyTorch docs: https://pytorch.org/docs/
