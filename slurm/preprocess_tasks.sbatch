#!/bin/bash
#SBATCH -J hbn-preprocess
#SBATCH -p general
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH -t 08:00:00
#SBATCH -o logs/preprocess_%j.out
#SBATCH -e logs/preprocess_%j.err

set -euo pipefail

# Usage:
# sbatch slurm/preprocess_tasks.sbatch
# Optional overrides:
# sbatch --export=ALL,RAW_DIR=/work/users/$USER/eeg-project/raw/ds005515,OUT_DIR=/work/users/$USER/eeg-project/task_gan_data,SEG_DUR=2.0,MAX_FILES= slurm/preprocess_tasks.sbatch

module purge
module load conda/latest

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
RAW_DIR="${RAW_DIR:-$PROJECT_DIR/raw/ds005515}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/task_gan_data}"
SEG_DUR="${SEG_DUR:-2.0}"
MAX_FILES="${MAX_FILES:-}"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUT_DIR"

cd "$PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg-gans

echo "[$(date)] Starting preprocessing"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "RAW_DIR=$RAW_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "SEG_DUR=$SEG_DUR"
echo "MAX_FILES=${MAX_FILES:-<none>}"

if [[ -n "$MAX_FILES" ]]; then
  python task_based_eeg_preprocessing.py "$RAW_DIR" "$OUT_DIR" \
    --segment-duration "$SEG_DUR" \
    --max-files-per-task "$MAX_FILES"
else
  python task_based_eeg_preprocessing.py "$RAW_DIR" "$OUT_DIR" \
    --segment-duration "$SEG_DUR"
fi

echo "[$(date)] Preprocessing complete"
