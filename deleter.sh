#!/bin/bash
#SBATCH --job-name=cleanup
#SBATCH --output=cleanup_%j.log
#SBATCH --time=02:00:00       # adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G              # adjust as needed

cd ~/nobackup/autodelete/fm-data-2

for dir in tomo-*; do
    rm -rf "$dir/tiling-2"
done