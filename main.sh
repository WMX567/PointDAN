#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=24:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --output=ms2ver13.out


source ~/.bashrc
conda activate py38



python train.py -source modelnet -target scannet  -scaler 1 -weight 0.5 -tb_log_dir ./logs/sa_m2ss

