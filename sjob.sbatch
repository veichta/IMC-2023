#!/bin/bash
#SBATCH --job-name="imc-train"
#SBATCH --mail-type=ALL
#SBATCH --mail-user="veichta@ethz.ch"

#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:14000m

#SBATCH --time=03:00:00

#SBATCH --output="logs/out.txt"

# run script
python main.py \
    --data /cluster/scratch/veichta/image-matching-challenge-2023/ \
    --output /cluster/scratch/veichta/outputs/ \
    --mode train \
    --config SIFT DISK ALIKED2K \
    --rotation_matching \
    --shared_camera \
    --retrieval netvlad \
    --pixsfm \
    --pixsfm_config ext_deps/pixel-perfect-sfm/pixsfm/configs/low_memory.yaml \
    --pixsfm_low_mem_threshold 5 \
    --overwrite
# --skip dioscuri wall kyiv-puppet-theater \
