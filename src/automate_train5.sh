
jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
export NUM_GPUS=2  # Set to equal gres=gpu:#!
export BATCH_SIZE=7 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export CUDA_VISIBLE_DEVICES="1,2,3"

for subj in 1; do
    pretrain_model_name="multisubject_subj0${subj}_hypatia_ip-adapter_sdxl_vit-h_finetune"
    echo pretrain_model_name=${pretrain_model_name}
    # accelerate launch --num_processes 2 --multi_gpu --mixed_precision "fp16" \
    # Train.py --data_path=../dataset \
    #     --cache_dir=../cache \
    #     --model_name=${pretrain_model_name} \
    #     --multi_subject \
    #     --subj=$subj \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_lr=6e-5 \
    #     --mixup_pct=.33 \
    #     --num_epochs=150 \
    #     --no-use_prior \
    #     --prior_scale=30 \
    #     --clip_scale=1 \
    #     --blur_scale=.5 \
    #     --recon_scale=10 \
    #     --no-use_image_aug \
    #     --n_blocks=4 \
    #     --hidden_dim=1024 \
    #     --num_sessions=40 \
    #     --ckpt_interval=999 \
    #     --ckpt_saving \
    #     --wandb_log \
    #     --pretrained_diffusion_model_path="stabilityai/stable-diffusion-xl-base-1.0" \
    #     --pretrained_ip_adapter_path="IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin" \
    #     --image_encoder_path="IP-Adapter/models/image_encoder" \
    #     --num_tokens=4 \
    #     --no-blurry_recon

    # # singlesubject finetuning
    model_name="pretrained_subj0${subj}_40sess_hypatia_ip-adapter_sdxl_vit-h_finetune"
    echo model_name=${model_name}
    accelerate launch --num_processes 2 --multi_gpu --mixed_precision "fp16" \
    Train.py \
        --data_path=../dataset \
        --cache_dir=../cache \
        --model_name=${model_name} \
        --no-multi_subject \
        --subj=$subj \
        --batch_size=${BATCH_SIZE} \
        --max_lr=6e-5 \
        --mixup_pct=.33 \
        --num_epochs=150 \
        --no-use_prior \
        --prior_scale=30 \
        --clip_scale=1 \
        --blur_scale=.5 \
        --recon_scale=10 \
        --no-use_image_aug \
        --n_blocks=4 \
        --hidden_dim=1024 \
        --num_sessions=40 \
        --ckpt_interval=999 \
        --ckpt_saving \
        --wandb_log \
        --pretrained_diffusion_model_path="stabilityai/stable-diffusion-xl-base-1.0" \
        --image_encoder_path="IP-Adapter/models/image_encoder" \
        --num_tokens=4 \
        --no-blurry_recon 
        #--resume_from_ckpt
        # --multisubject_ckpt=../train_logs/${pretrain_model_name} \

    for mode in "imagery" "vision"; do
        python recon_inference_mi.py \
            --model_name $model_name \
            --subj $subj \
            --mode $mode \
            --cache_dir ../cache \
            --data_path ../dataset \
            --hidden_dim 1024 \
            --n_blocks 4

        done
    done