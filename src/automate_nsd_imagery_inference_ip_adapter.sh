jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python
jupyter nbconvert Train.ipynb --to python
export CUDA_VISIBLE_DEVICES="0"

for subj in 1; do
    model_name="subj0${subj}_40sess_hypatia_ridge_sdxl_ip_adapter_plus"
    for mode in "vision" "imagery"; do # "vision" "imagery"; do

        # python recon_inference_mi.py \
        #     --model_name $model_name \
        #     --subj $subj \
        #     --mode $mode \
        #     --cache_dir ../cache \
        #     --data_path ../dataset \
        #     --hidden_dim 1024 \
        #     --n_blocks 4 \
        #     --dual_guidance

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
        subj01_40sess_hypatia_ridge_flat_dp5,
        subj01_40sess_hypatia_ridge_rank_order_rois_13,
        subj01_40sess_hypatia_ridge_sdxl_ip_adapter_plus"  \
        --data_path ../dataset \
        --output_path ../figs/ \
        --output_file methods_scatter_reduced2

done

