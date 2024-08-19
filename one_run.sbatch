#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --job-name=filtering
#SBATCH --cpus-per-task=2
#SBATCH --time=15:00:00


# use srun to launch Jupyter server in order to reserve a port
echo "hello"
conda activate client_selection

echo "print SLURM variables, hostnames and GPUs"
env | grep SLURM | while read line; do echo "# $line"; done
hostname
nvidia-smi -L

# Define a directory for job output
output_dir="$HOME/out_dir/$SLURM_JOB_ID"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

#HYDRA_FULL_ERROR=1 python3 main.py output_dir=/home/output_dir
python3 main.py gpus=[0] output_dir=??? bandit=$1 randomized=$2 datamodule.seed=$3 split.alpha=$4
echo "Waiting for experiments to finish"
wait
echo "Finished"