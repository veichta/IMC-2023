srun --time=10:00:00 --nodes=1 --ntasks=1 --cpus-per-task=20 --mem-per-cpu=3000 --gres=gpumem:14240m --gpus=rtx_3090:1 --pty bash
