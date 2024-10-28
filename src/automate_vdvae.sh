jupyter nbconvert Train_vdvae.ipynb --to python
jupyter nbconvert recon_inference_mi_vdvae.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="1"
subj=1 
model_name="subj01_40sess_hypatia_ridge_scv_0.70_strength_10000_wd_fs_fcon"

for mode in "vision" "imagery"; do #

    python recon_inference_mi_vdvae.py \
        --model_name $model_name \
        --subj $subj \
        --mode $mode \
        --cache_dir ../cache \
        --data_path ../dataset \
        --save_raw \
        --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
        --dual_guidance \
        --strength 0.70 \
        --filter_contrast \
        --filter_sharpness \
        
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


for weight_decay in 100000 1000000 10000000; do

    model_name="subj01_40sess_hypatia_ridge_scv_0.70_strength_${weight_decay}_wd_fs_fcon"
    echo model_name=${model_name}

    
    python Train_vdvae.py \
        --data_path=../dataset \
        --cache_dir=../cache \
        --model_name=${model_name} \
        --no-multi_subject \
        --subj=${subj} \
        --dual_guidance \
        --weight_decay=$weight_decay \

    for mode in "vision" "imagery"; do #

        python recon_inference_mi_vdvae.py \
            --model_name $model_name \
            --subj $subj \
            --mode $mode \
            --cache_dir ../cache \
            --data_path ../dataset \
            --save_raw \
            --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
            --dual_guidance \
            --strength 0.70 \
            --filter_contrast \
            --filter_sharpness \
            
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
    done

python plots_across_methods.py \
    --methods "mindeye1_subj01, \
    braindiffuser_subj01, \
    final_subj01_pretrained_40sess_24bs, \
    subj01_40sess_hypatia_ridge_sc3, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae, \
    subj01_40sess_hypatia_ridge_scv_0.70_strength,
    subj01_40sess_hypatia_ridge_scv_0.70_strength_10000_wd_fs_fcon, \
    subj01_40sess_hypatia_ridge_scv_0.70_strength_100000_wd_fs_fcon, \
    subj01_40sess_hypatia_ridge_scv_0.70_strength_1000000_wd_fs_fcon, \
    subj01_40sess_hypatia_ridge_scv_0.70_strength_10000000_wd_fs_fcon" \
    --data_path ../dataset \
    --output_path ../figs/ \
    --output_file methods_scatter_vdvae_weight_decay