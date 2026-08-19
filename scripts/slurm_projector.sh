#!/usr/bin/env bash
#SBATCH --job-name=graphmodi-projector
#SBATCH --output=outputs/slurm-%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG="${1:-configs/qwen8b_projector.yaml}"
mkdir -p outputs

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

uv run graph-modi train-projector --config "$CONFIG"
uv run graph-modi evaluate --config "$CONFIG"
