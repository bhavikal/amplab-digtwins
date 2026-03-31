#!/bin/bash
#SBATCH -J hbn-task-gan
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

set -euo pipefail

# Usage:
# sbatch slurm/train_task_gans.sbatch
# Optional overrides:
# sbatch --export=ALL,DATA_DIR=/work/users/$USER/eeg-project/task_gan_data,MODEL_DIR=/work/users/$USER/eeg-project/task_gan_models,EPOCHS=100,MODEL_PRESET=base,BATCH=32,MAX_SEGS= slurm/train_task_gans.sbatch
# Smoke test profile:
# sbatch --export=ALL,EPOCHS=10,MODEL_PRESET=small,BATCH=8,MAX_SEGS=1000 slurm/train_task_gans.sbatch

module purge
module load conda/latest

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/task_gan_data}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/task_gan_models}"
EPOCHS="${EPOCHS:-100}"
MODEL_PRESET="${MODEL_PRESET:-full}"
BATCH="${BATCH:-32}"
LATENT="${LATENT:-100}"
N_CRITIC="${N_CRITIC:-5}"
WCLIP="${WCLIP:-0.01}"
MAX_SEGS="${MAX_SEGS:-}"
SKIP_TASKS="${SKIP_TASKS:-}"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$MODEL_DIR"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg-gans

echo "[$(date)] Starting GAN training"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_DIR=$DATA_DIR"
echo "MODEL_DIR=$MODEL_DIR"
echo "EPOCHS=$EPOCHS"
echo "MODEL_PRESET=$MODEL_PRESET"
echo "BATCH=$BATCH"
echo "LATENT=$LATENT"
echo "N_CRITIC=$N_CRITIC"
echo "WCLIP=$WCLIP"
echo "MAX_SEGS=${MAX_SEGS:-<none>}"
echo "SKIP_TASKS=${SKIP_TASKS:-<none>}"

CMD=(
  python task_based_eeg_gan.py "$DATA_DIR" "$MODEL_DIR" "$EPOCHS"
  --device cuda
  --model-preset "$MODEL_PRESET"
  --batch-size "$BATCH"
  --latent-dim "$LATENT"
  --n-critic "$N_CRITIC"
  --weight-clip "$WCLIP"
)

if [[ -n "$MAX_SEGS" ]]; then
  CMD+=(--max-segments-per-task "$MAX_SEGS")
fi

if [[ -n "$SKIP_TASKS" ]]; then
  # shellcheck disable=SC2206
  TASKS=( $SKIP_TASKS )
  CMD+=(--skip-tasks "${TASKS[@]}")
fi

"${CMD[@]}"

echo "[$(date)] GAN training complete"
