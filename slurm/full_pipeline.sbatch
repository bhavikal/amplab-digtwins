#!/bin/bash
#SBATCH -J hbn-full-pipeline
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -o logs/full_%j.out
#SBATCH -e logs/full_%j.err

set -euo pipefail

# Usage:
# sbatch slurm/full_pipeline.sbatch
# Example smoke test:
# sbatch --export=ALL,EPOCHS=10,MODEL_PRESET=small,BATCH=8,MAX_FILES=3,MAX_SEGS=1000 slurm/full_pipeline.sbatch

module purge
module load conda/latest

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
RAW_DIR="${RAW_DIR:-$PROJECT_DIR/raw/ds005515}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/task_gan_data}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/task_gan_models}"
SEG_DUR="${SEG_DUR:-2.0}"
MAX_FILES="${MAX_FILES:-}"
EPOCHS="${EPOCHS:-100}"
MODEL_PRESET="${MODEL_PRESET:-full}"
BATCH="${BATCH:-32}"
MAX_SEGS="${MAX_SEGS:-}"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$DATA_DIR" "$MODEL_DIR"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg-gans

echo "[$(date)] Step 1/2: Preprocessing"
if [[ -n "$MAX_FILES" ]]; then
  python task_based_eeg_preprocessing.py "$RAW_DIR" "$DATA_DIR" \
    --segment-duration "$SEG_DUR" \
    --max-files-per-task "$MAX_FILES"
else
  python task_based_eeg_preprocessing.py "$RAW_DIR" "$DATA_DIR" \
    --segment-duration "$SEG_DUR"
fi

echo "[$(date)] Step 2/2: GAN training"
CMD=(
  python task_based_eeg_gan.py "$DATA_DIR" "$MODEL_DIR" "$EPOCHS"
  --device cuda
  --model-preset "$MODEL_PRESET"
  --batch-size "$BATCH"
)

if [[ -n "$MAX_SEGS" ]]; then
  CMD+=(--max-segments-per-task "$MAX_SEGS")
fi

"${CMD[@]}"

echo "[$(date)] Full pipeline complete"
