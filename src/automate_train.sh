jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="0"

subj=1 

model_name="subj01_40sess_hypatia_ridge_sc3"
echo model_name=${model_name}

# python Train.py \
#     --data_path=../dataset \
#     --cache_dir=../cache \
#     --model_name=${model_name} \
#     --no-multi_subject \
#     --subj=${subj} \
#     --weight_decay=60000 \
#     --dual_guidance

for mode in "vision" "imagery"; do

    # python recon_inference_mi.py \
    #     --model_name $model_name \
    #     --subj $subj \
    #     --mode $mode \
    #     --cache_dir ../cache \
    #     --data_path ../dataset \
    #     --save_raw \
    #     --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
    #     --dual_guidance
        
    python final_evaluations_mi_multi.py \
            --model_name $model_name \
            --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
            --subj $subj \
            --mode $mode \
            --data_path ../dataset \
            --cache_dir ../cache
            # --no-blurry_recon

    # python plots_across_subjects.py \
    #         --model_name="${model_name}" \
    #         --mode="${mode}" \
    #         --data_path ../dataset \
    #         --cache_dir ../cache \
    #         --criteria all \
    #         --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
    #         --subjs=$subj

    done

python plots_across_methods.py \
    --methods "mindeye1_subj01, \
    braindiffuser_subj01, \
    final_subj01_pretrained_40sess_24bs, \
    pretrained_subj01_40sess_hypatia_vd2, \
    pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
    subj01_40sess_hypatia_turbo_ridge_flat3, \
    subj01_40sess_hypatia_ridge_flat_dp5, \
    subj01_40sess_hypatia_ridge_rank_order_rois_13, \
    subj01_40sess_hypatia_ridge_sc3" \
    --data_path ../dataset \
    --output_path ../figs/ \
    --output_file methods_scatter_reduced3