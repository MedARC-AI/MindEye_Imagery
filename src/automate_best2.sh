jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="0"
for subj in 1; do
    model_name="subj0${subj}_40sess_hypatia_mirage2"

    for mode in "imagery"; do #
            
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
#     subj01_40sess_hypatia_ridge_sc3, \
#     subj01_40sess_hypatia_ridge_scv_0.70_strength, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_medium_captions" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --output_file methods_scatter_best