#!/bin/sh

#SBATCH --job-name=actlearn

#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --partition=long

source ~/venv/bin/activate

python main.py "$@"
