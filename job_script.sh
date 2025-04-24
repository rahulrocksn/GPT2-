#!/bin/bash
#SBATCH -t 04:00:00               # Time limit: 4 hours
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=8                # Number of tasks (parallel threads)
#SBATCH --gpus-per-node=1         # Number of GPUs per node
#SBATCH -A standby                # Account/Partition: standby
#SBATCH --output=train_output2.log  # Standard output log
#SBATCH --error=train_error2.log    # Error log


module load conda/2024.09
module load cuda

# Activate the virtual environment
cd /home/rnahar/Desktop/CS587
source 587/bin/activate

# Change to the directory where hw1_training.py is located
cd /home/rnahar/Desktop/another

# Run the training script
python train.py