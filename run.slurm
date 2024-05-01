#!/bin/bash
#SBATCH --partition=gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=python-gpu
#SBATCH --output=/scratch/%u/Project/outputs/%x-%N-%j.out   # Output file
#SBATCH --error=/scratch/%u/Project/errors/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4                # number of cores needed
#SBATCH --gres=gpu:3g.40gb:1                # up to 8; only request what you need
#SBATCH --mem-per-cpu=4000M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL 
#SBATCH --time=0-04:00:00                   # D-HH:MM:SS; please choose carefully

set -x
umask 0027

# change directory to the root of the project
cd /scratch/apathak2/Project

# Add the current directory (project root) to the PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10                           
module load python

python src/main.py
