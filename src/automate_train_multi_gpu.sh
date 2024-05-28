
jupyter nbconvert Train.ipynb --to python
export NUM_GPUS=3  # Set to equal gres=gpu:#!
export BATCH_SIZE=21 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export CUDA_VISIBLE_DEVICES="0,2,3"

export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export HOSTNAMES=$(hostname)
export MASTER_ADDR=$(hostname)
export COUNT_NODE=1

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

pretrain_model_name="multisubject_subj01_hypatia_contrastive"
echo pretrain_model_name=${pretrain_model_name}
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=../dataset --cache_dir=../cache --model_name=${pretrain_model_name} --multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --no-blurry_recon

# singlesubject finetuning
model_name="pretrained_subj01_40sess_hypatia_contrastive"
echo model_name=${model_name}
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=../dataset --cache_dir=../cache --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-5 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --multisubject_ckpt=../train_logs/${pretrain_model_name} --no-blurry_recon