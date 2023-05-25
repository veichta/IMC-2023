export INSTALL_DIR=$HOME/.local/

echo "load gcc and python"
module load gcc/8.2.0
module load python_gpu/3.10.4 cuda/11.8.0 cudnn/8.8.1.3
module load eth_proxy


if [ -f /cluster/scratch/$USER/IMC/bin/activate ];
then
    source /cluster/scratch/$USER/IMC/bin/activate;
fi

