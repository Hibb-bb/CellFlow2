#!/usr/bin/env bash
#SBATCH --job-name=cellflow
#SBATCH --output=cellflow-%j.out
#SBATCH --error=cellflow-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00

set -euo pipefail

echo "[$(date)] Starting job ${SLURM_JOB_ID:-N/A} on $(hostname)"

# Always start from the directory where sbatch was submitted
cd "${SLURM_SUBMIT_DIR}"

REPO_ROOT="${SLURM_SUBMIT_DIR}/CellFlow2"
VENV_DIR="${REPO_ROOT}/.venv"

if [[ ! -d "$REPO_ROOT" ]]; then
    echo "ERROR: repo not found: $REPO_ROOT" >&2
    exit 1
fi

cd "$REPO_ROOT"

module load uv
module load cuda/12.6.1_560.35.03
# Create the environment only if it does not already exist
if [[ ! -d "$VENV_DIR" ]]; then
    uv venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"

# Install project into the venv
uv pip install -e .

# Optional: only keep this if your cluster actually supports CUDA 12 wheels for JAX
uv pip install -U "jax[cuda12]"

# Runtime dependency used by cellflow.preprocessing._gene_emb
uv pip install requests

# Installing rapids-singlecell

# RAPIDS single-cell stack for CUDA 12
# Install exactly ONE RAPIDS single-cell stack for CUDA 12
uv pip install --pre \
  --extra-index-url=https://pypi.nvidia.com \
  "rapids-singlecell-cu12[rapids]"

# Sanity checks before the real job
python - <<'PY'
import requests
import cuml
import rapids_singlecell
import cellflow
print("requests OK")
print("cuml OK")
print("rapids_singlecell OK")
print("cellflow OK")
PY


echo "Python: $(which python)"
python --version

cd docs/notebooks

# Use srun for the actual job payload
uv run python3 zb_fish.py

echo "[$(date)] Job finished"
