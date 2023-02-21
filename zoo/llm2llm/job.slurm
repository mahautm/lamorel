#!/bin/bash
#SBATCH --job-name=test-lamorel
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=alien
#SBATCH --qos=alien
#SBATCH --gres=gpu:5

pwd
srun /homedtcl/mmahaut/projects/lamorel/zoo/llm2llm/launcher.sh \
    rl_script_args.path=/homedtcl/mmahaut/projects/lamorel/zoo/llm2llm/train.py \
    --config-path=/homedtcl/mmahaut/projects/lamorel/zoo/llm2llm/configs/ \
    --config-name=local_gpu_config