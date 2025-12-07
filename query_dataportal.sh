#!/bin/bash
#SBATCH --job-name=querydataportal
#SBATCH --output=querydataportal_%j.log
#SBATCH --time=04:00:00       # adjust as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G              # adjust as needed

source .venv/bin/activate
python query_dataportal.py