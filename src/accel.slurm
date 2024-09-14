#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=p5
#SBATCH --job-name=eye
#SBATCH --nodes=1           
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue

source ~/.bashrc
mamba init
mamba activate imagery2
cd /weka/proj-fmri/jonxu/MindEye_Imagery/src
jupyter nbconvert Train.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=200 # 400 and 200 # (orig. paper used 42 for multisubject / 24 for singlesubject). 48gb vram
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

# multisubject pretraining
model_name="ip_adapt_finetune_subj001"
echo model_name=${model_name}s
#accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=/weka/proj-medarc/shared/mindeyev2_dataset --cache_dir=/weka/proj-medarc/shared/cache --model_name=${model_name} --multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log

#python Train.py --data_path=/weka/proj-medarc/shared/mindeyev2_dataset --cache_dir=/weka/proj-medarc/shared/cache --model_name=${model_name} --multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=20 --ckpt_saving --wandb_log

# singlesubject finetuning
#model_name="pretrained_subj01_40sess_ext1"
#echo model_name=${model_name}
#accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=/weka/proj-medarc/shared/mindeyev2_dataset --cache_dir=/weka/proj-medarc/shared/cache --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/multisubject_subj01_ext1

python Train.py --data_path=/weka/proj-medarc/shared/mindeyev2_dataset --cache_dir=/weka/proj-medarc/shared/cache --model_name={model_name} --no-multi_subject --subj=1 --batch_size={batch_size} --num_sessions=40 --hidden_dim=1024 --clip_scale=1. --blurry_recon --blur_scale=.5 --seq_past=0 --seq_future=0 --use_prior --prior_scale=30 --n_blocks=4 --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --no-use_image_aug --ckpt_interval=20 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/ip_adapt_multisubject_excl01"
