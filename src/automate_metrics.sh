jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="0"

subj=1 
for model_name in "mindeye1_subj01" "braindiffuser_subj01" "pretrained_subj01_40sess_hypatia_vd2" "pretrained_subj01_40sess_hypatia_vd_dual_proj_avg" "subj01_40sess_hypatia_turbo_ridge_flat3" "subj01_40sess_hypatia_ridge_sc3"; do
    echo model_name=${model_name}

    for mode in "vision" "imagery"; do
            
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

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     pretrained_subj01_40sess_hypatia_vd2, \
#     pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
#     subj01_40sess_hypatia_turbo_ridge_flat3, \
#     subj01_40sess_hypatia_ridge_flat_dp5, \
#     subj01_40sess_hypatia_ridge_rank_order_rois_13, \
#     subj01_40sess_hypatia_ridge_sc3,
#     subj01_40sess_hypatia_ridge_sc_flux_enhanced" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --output_file methods_scatter_reduced3