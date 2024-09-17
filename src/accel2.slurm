#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=p5
#SBATCH --job-name=ses20
#SBATCH --nodes=1           
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue
#SBATCH --qos=idle

# Make sure you activate your fmri environment created from src/setup.sh
cd /weka/proj-fmri/ckadirt/MindEye_Imagery/src
source /admin/home-ckadirt/fmri/bin/activate

# The following line converts your jupyter notebook into a python script runnable with Slurm
jupyter nbconvert Train_vd.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=63 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

subj=1
num_sessions=20
pretrain_model_name="multisubject_subj0${subj}_hypatia_vd2_sessions${num_sessions}"
echo model_name=${pretrain_model_name}
python Train_vd.py --data_path=/weka/proj-medarc/shared/mindeyev2_dataset --cache_dir=/weka/proj-medarc/shared/cache --model_name=${pretrain_model_name} --multi_subject --subj=${subj} --batch_size=${BATCH_SIZE} --max_lr=3e-5 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=${num_sessions} --ckpt_interval=999 --ckpt_saving --wandb_log #--no-blurry_recon


# singlesubject finetuning
model_name="pretrained_subj0${subj}_40sess_hypatia_vd2_sessions${num_sessions}"
echo model_name=${model_name}
python Train_vd.py --data_path=/weka/proj-medarc/shared/mindeyev2_dataset --cache_dir=/weka/proj-medarc/shared/cache --model_name=${model_name} --no-multi_subject --subj=${subj} --batch_size=${BATCH_SIZE} --max_lr=3e-5 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=${num_sessions} --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/${pretrain_model_name} #--no-blurry_recon #--resume_from_ckpt

# jupyter nbconvert recon_inference_mi_vd.ipynb --to python
# for mode in "imagery" "vision"; do
#     python recon_inference_mi_vd.py \
#         --model_name $model_name \
#         --subj $subj \
#         --mode $mode \
#         --cache_dir ../cache \
#         --data_path ../dataset \
#         --hidden_dim 1024 \
#         --n_blocks 4

#     done



# multisubject pretraining
#model_name="multisubject_excludingsubj01_40sess"
#echo model_name=${model_name}
#accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=/weka/proj-fmri/shared/mindeyev2_dataset --cache_dir=/weka/proj-fmri/shared/cache --model_name=${model_name} --multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log

# singlesubject finetuning
#model_name="finetuned_subj01_40sess"
#echo model_name=${model_name}
#accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=/weka/proj-fmri/shared/mindeyev2_dataset --cache_dir=/weka/proj-fmri/shared/cache --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/multisubject_excludingsubj01_40sess