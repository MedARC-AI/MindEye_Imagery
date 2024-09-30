# Write all the code in jupyter notebook then covert the file.
jupyter nbconvert Train.ipynb --to python
export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=42 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export CUDA_VISIBLE_DEVICES="2" # Set's the GPU device

subj=1 
pretrain_model_name="multisubject_subj0${subj}_hypatia_vd_snr_0_6"
# echo model_name=${pretrain_model_name}
python Train.py --data_path=../dataset --cache_dir=../cache --model_name=${pretrain_model_name} --multi_subject --subj=${subj} --batch_size=${BATCH_SIZE} --max_lr=3e-5 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --snr_threshold=.60 #--resume_from_ckpt #--no-blurry_recon

# export BATCH_SIZE=84
# singlesubject finetuning
model_name="pretrained_subj0${subj}_40sess_hypatia_vd_multisubject_snr_0_6"
echo model_name=${model_name}
python Train.py --data_path=../dataset --cache_dir=../cache --model_name=${model_name} --no-multi_subject --subj=${subj} --batch_size=${BATCH_SIZE} --max_lr=3e-5 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/${pretrain_model_name} --snr_threshold=.60 #--resume_from_ckpt  #--no-blurry_recon --resume_from_ckpt

jupyter nbconvert recon_inference_mi.ipynb --to python
for mode in "imagery" "vision"; do
    python recon_inference_mi.py \
        --model_name $model_name \
        --subj $subj \
        --mode $mode \
        --cache_dir ../cache \
        --data_path ../dataset \
        --hidden_dim 1024 \
        --n_blocks 4 \
        --snr 0.6

    python final_evaluations_mi_multi.py \
        --model_name $model_name \
        --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
        --subj $subj \
        --mode $mode \
        --data_path ../dataset \
        --cache_dir ../cache
        # --no-blurry_recon

    python plots_across_subjects.py \
            --model_name="${model_name}" \
            --mode="${mode}" \
            --data_path ../dataset \
            --cache_dir ../cache \
            --criteria all \
            --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
            --subjs=$subj

    done

python plots_across_methods.py \
--methods "mindeye1_subj01,final_subj01_pretrained_40sess_24bs,pretrained_subj01_40sess_hypatia_ip_adapter_plus,pretrained_subj01_40sess_hypatia_ip_adapter2,pretrained_subj01_40sess_hypatia_vd2,pretrained_subj01_40sess_hypatia_vd_dual_proj,pretrained_subj01_40sess_hypatia_vd_snr_0_5,pretrained_subj01_40sess_hypatia_vd_snr_0_55,pretrained_subj01_40sess_hypatia_vd_snr_0_65,pretrained_subj01_40sess_hypatia_vd_snr_0_75,pretrained_subj01_40sess_hypatia_vd_multisubject_snr_0_5,pretrained_subj01_40sess_hypatia_vd_multisubject_snr_0_6" \
--data_path ../dataset \
--output_path ../figs/

