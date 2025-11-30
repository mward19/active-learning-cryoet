#!/bin/bash
#SBATCH --job-name=tomo_tiles
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-1558
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source .venv/bin/activate

python dataset_creation.py --index $SLURM_ARRAY_TASK_ID