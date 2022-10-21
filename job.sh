#!/bin/bash
#YBATCH -r am_1
#SBATCH --nodes 1
#SBATCH -J resnet
#SBATCH --time=10:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err 

. /etc/profile.d/modules.sh
module load cuda
module load cudnn
module load openmpi

wandb agent riverstone/auto_interval/eoh19u3o