jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="1"

for subj in 3 4; do
    for weight_decay in 50000 100000 250000 500000 750000 1000000; do

        model_name="subj0${subj}_40sess_hypatia_ridge_scv_0.70_strength_${weight_decay}_wd_fs_fcon"
        echo model_name=${model_name}

        
        python Train.py \
            --data_path=../dataset \
            --cache_dir=../cache \
            --model_name=${model_name} \
            --no-multi_subject \
            --subj=${subj} \
            --dual_guidance \
            --weight_decay=$weight_decay \

        for mode in "vision" "imagery"; do #

            python recon_inference_mi.py \
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
    done

python plots_across_methods.py \
    --methods "subj03_40sess_hypatia_ridge_scv_0.70_strength_50000_wd_fs_fcon, \
    subj03_40sess_hypatia_ridge_scv_0.70_strength_100000_wd_fs_fcon, \
    subj03_40sess_hypatia_ridge_scv_0.70_strength_250000_wd_fs_fcon, \
    subj03_40sess_hypatia_ridge_scv_0.70_strength_500000_wd_fs_fcon, \
    subj03_40sess_hypatia_ridge_scv_0.70_strength_750000_wd_fs_fcon, \
    subj03_40sess_hypatia_ridge_scv_0.70_strength_1000000_wd_fs_fcon" \
    --data_path ../dataset \
    --output_path ../figs/ \
    --output_file methods_scatter_vdvae_weight_decay
    