jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=35 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export CUDA_VISIBLE_DEVICES="2"

subj=1 
pretrain_model_name="multisubject_subj0${subj}_hypatia_new_vd_dual_proj"
echo model_name=${pretrain_model_name}
# python Train.py \
#     --data_path=../dataset \
#     --cache_dir=../cache \
#     --model_name=${pretrain_model_name} \
#     --multi_subject --subj=${subj} \
#     --batch_size=${BATCH_SIZE} \
#     --max_lr=3e-5 \
#     --mixup_pct=.33 \
#     --num_epochs=150 \
#     --use_prior \
#     --prior_scale=30 \
#     --clip_scale=1 \
#     --blur_scale=.5 \
#     --no-use_image_aug \
#     --n_blocks=4 \
#     --hidden_dim=1024 \
#     --num_sessions=40 \
#     --ckpt_interval=999 \
#     --ckpt_saving \
#     --wandb_log \
#     --dual_guidance

# singlesubject finetuning

model_name="pretrained_subj0${subj}_40sess_hypatia_new_vd_dual_proj"
echo model_name=${model_name}
python Train.py --data_path=../dataset \
    --cache_dir=../cache \
    --model_name=${model_name} \
    --no-multi_subject \
    --subj=${subj} \
    --batch_size=${BATCH_SIZE} \
    --max_lr=3e-5 \
    --mixup_pct=.33 \
    --num_epochs=150 \
    --use_prior \
    --prior_scale=30 \
    --clip_scale=1 \
    --blur_scale=.5 \
    --no-use_image_aug \
    --n_blocks=4 \
    --hidden_dim=1024 \
    --num_sessions=40 \
    --ckpt_interval=999 \
    --ckpt_saving \
    --wandb_log \
    --multisubject_ckpt=../train_logs/${pretrain_model_name} \
    --dual_guidance 


for mode in "imagery" "vision"; do
    python recon_inference_mi.py \
        --model_name $model_name \
        --subj $subj \
        --mode $mode \
        --cache_dir ../cache \
        --data_path ../dataset \
        --hidden_dim 1024 \
        --n_blocks 4 \
        --dual_guidance

    python final_evaluations_mi_multi.py \
            --model_name $model_name \
            --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
            --subj $subj \
            --mode $mode \
            --data_path ../dataset \
            --cache_dir ../cache

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
--methods "mindeye1_subj01, \
braindiffuser_subj01, \
final_subj01_pretrained_40sess_24bs, \
pretrained_subj01_40sess_hypatia_vd2, \
pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
subj01_40sess_hypatia_turbo_ridge_flat3,
pretrained_subj01_40sess_hypatia_new_vd_dual_proj" \
--data_path ../dataset \
--output_path ../figs/ \
--output_file methods_scatter_new_vd_main