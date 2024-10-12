
jupyter nbconvert Train_ridge.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=50 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export CUDA_VISIBLE_DEVICES="1"

subj=1 

model_name="subj0${subj}_40sess_hypatia_ridge_flat_dp4"
echo model_name=${model_name}

for mode in "imagery"; do #"vision"
    python recon_inference_mi.py \
        --model_name $model_name \
        --subj $subj \
        --mode $mode \
        --cache_dir ../cache \
        --data_path ../dataset \
        --use_prior \
        --save_raw \
        --dual_guidance #\
        # --no-normalize_preds

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
--methods "mindeye1_subj01, \
braindiffuser_subj01, \
final_subj01_pretrained_40sess_24bs, \
pretrained_subj01_40sess_hypatia_vd2, \
pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
subj01_40sess_hypatia_turbo_ridge_flat3, \
subj0${subj}_40sess_hypatia_ridge_flat_dp2" \
--data_path ../dataset \
--output_path ../figs/ \
--output_file methods_scatter_reduced.png
