#!/usr/bin/env bash
# Wrapper: run Cell-JEPA pretrain then perturbation JEPA pretrain
# in the SAME Slurm allocation (no nested sbatch).
#
# Usage:
#   sbatch examples/pretraining/jepa/hidden_pretrain_ds_jepa_bruno_then_perturb.slurm
#
# Optional overrides:
#   STORAGE_ROOT, SAVE_ROOT, QUERY_NAME, DATASET, VOCAB_PATH
#   PERTURB_NPROC=1           # override GPU count for perturb stage
#   PERTURB_CUDA_VISIBLE_DEVICES=0

#SBATCH --job-name=bruno-cellflow
#SBATCH --output=bruno_cellflow-%j.out
#SBATCH --error=bruno_cellflow-%j.err
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --partition=gpu
#SBATCH --mem=128G

set -euo pipefail


git clone https://github.com/Hibb-bb/CellFlow2.git


cd CellFlow2

uv init

uv venv

source ./.venv/bin/activateß

uv pip install -e .

uv pip install -U "jax[cuda12]"


cd docs/notebooks

uv run python3 zb_fish.py

# # Resolve repo root safely (avoid /var/spool when sbatch stages the script)
# SCRIPT_DIR_RAW="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR_RAW}/../../.." && pwd)"
# REPO_ROOT=${REPO_ROOT:-${REPO_ROOT_DEFAULT}}
# if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
#   if [ -f "${SLURM_SUBMIT_DIR}/paths_bruno.sh" ] || [ -d "${SLURM_SUBMIT_DIR}/data/cellxgene" ]; then
#     REPO_ROOT="${SLURM_SUBMIT_DIR}"
#   fi
# fi
# if [ ! -d "$REPO_ROOT" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
#   REPO_ROOT="${SLURM_SUBMIT_DIR}"
# fi

# cd "$REPO_ROOT"
# mkdir -p slurm-logs

# # Load Bruno paths only (no fallback).
# if [ -f "${REPO_ROOT}/paths_bruno.sh" ]; then
#   # shellcheck disable=SC1091
#   . "${REPO_ROOT}/paths_bruno.sh"
# else
#   echo "ERROR: paths_bruno.sh not found at ${REPO_ROOT}/paths_bruno.sh" >&2
#   exit 1
# fi

# export SLURM_CPU_BIND="cores"
# export WANDB_ENTITY="${WANDB_ENTITY:-a10v-1}"

# # ---- Stage 1: Cell-JEPA pretrain (blood) ----
# export QUERY_NAME="${QUERY_NAME:-blood}"
# export STORAGE_ROOT="${STORAGE_ROOT:-/hpc/mydata/ywen/cell-jepa-data}"
# export SAVE_ROOT="${SAVE_ROOT:-/hpc/mydata/ywen/cell-jepa-saves}"
# export DATASET="${DATASET:-${STORAGE_ROOT}/cellxgene/scb/${QUERY_NAME}/all_counts}"
# export VOCAB_PATH="${VOCAB_PATH:-${STORAGE_ROOT}/cellxgene/vocab/default_census_vocab.json}"

# if [ ! -d "${DATASET}" ]; then
#   echo "ERROR: DATASET directory not found: ${DATASET}" >&2
#   exit 1
# fi

# export VENV_ACTIVATE="${VENV_ACTIVATE:-${REPO_ROOT}/.venv-ubuntu-nvidia/bin/activate}"
# # ---- VICReg-cov emergency fuse (salvage) ----
# # Set VICREG_COV_FUSE=0 to disable.
# export VICREG_COV_FUSE="${VICREG_COV_FUSE:-1}"
# export VICREG_COV_FUSE_MAX_POST="${VICREG_COV_FUSE_MAX_POST:-5}"
# # Explicit mask toggle (default OFF for wrapper)
# export EXPLICIT_MASK="${EXPLICIT_MASK:-0}"
# # ---- Batch sizing (Stage 1) ----
# export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
# export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
# # ---- Stage 1 model overrides ----
# export JEPA_WEIGHT="${JEPA_WEIGHT:-1.0}"
# export JEPA_TEACHER_BLEND="${JEPA_TEACHER_BLEND:-final}"
# export NO_CCE="${NO_CCE:-1}"
# export VICREG="${VICREG:-0}"
# export VICREG_VAR_WEIGHT="${VICREG_VAR_WEIGHT:-0.0}"
# export VICREG_COV_WEIGHT="${VICREG_COV_WEIGHT:-0.0}"
# echo "=== Stage 1: Cell-JEPA (blood) ==="
# echo "DATASET=${DATASET}"
# echo "VICREG_COV_FUSE=${VICREG_COV_FUSE} VICREG_COV_FUSE_MAX_POST=${VICREG_COV_FUSE_MAX_POST}"
# echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
# echo "JEPA_WEIGHT=${JEPA_WEIGHT} JEPA_TEACHER_BLEND=${JEPA_TEACHER_BLEND} NO_CCE=${NO_CCE}"
# echo "VICREG=${VICREG} VICREG_VAR_WEIGHT=${VICREG_VAR_WEIGHT} VICREG_COV_WEIGHT=${VICREG_COV_WEIGHT}"
# echo "EXPLICIT_MASK=${EXPLICIT_MASK}"
# bash "${REPO_ROOT}/examples/pretraining/jepa/hidden_pretrain_ds_jepa_bruno.slurm"

# # ---- Stage 2: Perturbation JEPA pretrain ----
# # Optional: restrict GPUs for perturbation stage
# if [ -n "${PERTURB_CUDA_VISIBLE_DEVICES:-}" ]; then
#   export CUDA_VISIBLE_DEVICES="${PERTURB_CUDA_VISIBLE_DEVICES}"
# fi
# if [ -n "${PERTURB_NPROC:-}" ]; then
#   export NPROC="${PERTURB_NPROC}"
# fi

# export VENV_ACTIVATE="${REPO_ROOT}/envs/perturbation/.venv-perturb/bin/activate"
# # ---- Batch sizing (Stage 2) ----
# export BATCH_SIZE="${BATCH_SIZE:-64}"
# export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
# # ---- Stage 2 model overrides ----
# export VICREG="${VICREG:-0}"
# export VICREG_VAR_WEIGHT="${VICREG_VAR_WEIGHT:-0.0}"
# export VICREG_COV_WEIGHT="${VICREG_COV_WEIGHT:-0.0}"
# echo "=== Stage 2: Perturbation JEPA ==="
# echo "BATCH_SIZE=${BATCH_SIZE} EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
# echo "VICREG=${VICREG} VICREG_VAR_WEIGHT=${VICREG_VAR_WEIGHT} VICREG_COV_WEIGHT=${VICREG_COV_WEIGHT}"
# bash "${REPO_ROOT}/examples/perturbation_pretraining/perturbation_pretrain_jepa_bruno.slurm"