#!/bin/bash
#SBATCH --output=logs/slurm_%j.log
#SBATCH --constraint=v100-32g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --account=krk@v100

echo $1
source config.sh
python python_scripts/train_model.py $1 $CAPS_DIR $PARTICIPANTS_TSV_FOLDER