#!/bin/bash
#SBATCH -J m3f
#SBATCH -o run_scripts/screen/%x-%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=160G

set -euo pipefail

# Activate the desired environment before submitting this script.
bash run_scripts/train.sh
