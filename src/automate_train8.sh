
jupyter nbconvert Train_ridge.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=50 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export CUDA_VISIBLE_DEVICES="2"

subj=1 

model_name="subj0${subj}_40sess_hypatia_turbo_ridge"
echo model_name=${model_name}
# python Train_ridge.py --data_path=../dataset --cache_dir=../cache --model_name=${model_name} --no-multi_subject --subj=${subj} --batch_size=${BATCH_SIZE} 

jupyter nbconvert recon_inference_mi_ridge.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python
for alpha in 10 100 1000 10000 100000; do #60000 0.01 0.1 1 
    alpha_model_name="${model_name}_alpha_${alpha}"
    for mode in "vision" "imagery"; do
        python recon_inference_mi_ridge.py \
            --model_name $alpha_model_name \
            --subj $subj \
            --mode $mode \
            --cache_dir ../cache \
            --data_path ../dataset \
            --alpha $alpha

        python final_evaluations_mi_multi.py \
                --model_name $alpha_model_name \
                --all_recons_path evals/${alpha_model_name}/${alpha_model_name}_all_recons_${mode}.pt \
                --subj $subj \
                --mode $mode \
                --data_path ../dataset \
                --cache_dir ../cache
                # --no-blurry_recon

        python plots_across_subjects.py \
                --model_name="${alpha_model_name}" \
                --mode="${mode}" \
                --data_path ../dataset \
                --cache_dir ../cache \
                --criteria all \
                --all_recons_path evals/${alpha_model_name}/${alpha_model_name}_all_recons_${mode}.pt \
                --subjs=$subj

        done
    done

python plots_across_methods.py \
--methods "mindeye1_subj01, \
braindiffuser_subj01, \
final_subj01_pretrained_40sess_24bs, \
pretrained_subj01_40sess_hypatia_vd2, \
pretrained_subj01_40sess_hypatia_vd_dual_proj, \
pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
pretrained_subj01_40sess_hypatia_ip_adapter2, \
pretrained_subj01_40sess_hypatia_ip_adapter_plus, \
subj01_40sess_hypatia_turbo_ridge_alpha_60000, \
subj01_40sess_hypatia_turbo_ridge_alpha_0.01, \
subj01_40sess_hypatia_turbo_ridge_alpha_0.1, \
subj01_40sess_hypatia_turbo_ridge_alpha_1, \
subj01_40sess_hypatia_turbo_ridge_alpha_10, \
subj01_40sess_hypatia_turbo_ridge_alpha_100, \
subj01_40sess_hypatia_turbo_ridge_alpha_1000, \
subj01_40sess_hypatia_turbo_ridge_alpha_10000, \
subj01_40sess_hypatia_turbo_ridge_alpha_100000" \
--data_path ../dataset \
--output_path ../figs/ \
--output_file methods_comparison_ridge_alpha