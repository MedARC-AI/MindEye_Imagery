jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="0"

subj=1 
for num_rois in {10..23}; do 
    model_name="subj0${subj}_40sess_hypatia_ridge_rank_order_rois_${num_rois}"
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

python plots_across_methods.py \
--methods "subj01_40sess_hypatia_nsd_general, \
mindeye1_subj01, \
braindiffuser_subj01, \
final_subj01_pretrained_40sess_24bs, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_10, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_11, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_12, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_13, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_14, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_15, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_16, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_17, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_18, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_19, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_20, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_21, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_22, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_23, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_24, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_25, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_26, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_27, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_28, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_29, \
subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30" \
--data_path ../dataset \
--output_path ../figs/ \
--output_file methods_comparison_roi_threshold_s1_samplewise_sc \
--gradient