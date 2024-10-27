jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="1"

subj=1 

for max_iter in 50000; do

    for weight_decay in 5000000 10000000; do

        model_name="subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_${weight_decay}_wd_${max_iter}_max_it"
        echo model_name=${model_name}

        python Train.py \
            --data_path=../dataset \
            --cache_dir=../cache \
            --model_name=${model_name} \
            --no-multi_subject \
            --subj=${subj} \
            --weight_decay=$weight_decay \
            --max_iter=$max_iter \
            --dual_guidance \

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
                --strength 0.7 \
                --vdvae
                
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

for subj in 2 5 7; do

    for weight_decay in 100000; do

        model_name="subj0${subj}_40sess_hypatia_ridge_sc_vdvae_0.70_strength_${weight_decay}_wd_50000_max_it"
        echo model_name=${model_name}

        python Train.py \
            --data_path=../dataset \
            --cache_dir=../cache \
            --model_name=${model_name} \
            --no-multi_subject \
            --subj=${subj} \
            --weight_decay=$weight_decay \
            --dual_guidance \

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
                --strength 0.7 \
                --vdvae
                
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
    --methods "mindeye1_subj01, \
    braindiffuser_subj01, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae, \
    subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_100000_wd_50000_max_it, \
    subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_500000_wd_50000_max_it, \
    subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_1000000_wd_50000_max_it, \
    subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_100000_wd_200000_max_it, \
    subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_500000_wd_200000_max_it, \
    subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_1000000_wd_200000_max_it, \
    subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_100000_wd_1000000_max_it, \
    subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_500000_wd_1000000_max_it, \
    subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_1000000_wd_1000000_max_it, \
    subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_5000000_wd_50000_max_it, \
    subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_10000000_wd_50000_max_it" \
    --data_path ../dataset \
    --output_path ../figs/ \
    --output_file methods_scatter_reduced4

# subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_${weight_decay}_wd_${max_iter}_max_it