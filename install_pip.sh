#!/usr/bin/env bash
set -e
source startup.sh


mkdir -p /cluster/scratch/$USER/IMC
echo "Creating virtual environment"
python -m venv /cluster/scratch/$USER/IMC
echo "Activating virtual environment"

source /cluster/scratch/$USER/IMC/bin/activate
/cluster/scratch/$USER/IMC/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#/cluster/scratch/$USER/IMC/bin/pip install -r requirements.txt

