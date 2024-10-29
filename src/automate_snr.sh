jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="3"
subj=3
model_name="subj03_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_snr_0.70"
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
        --snr_threshold 0.70
        
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

for subj in 3; do
    for snr in 0.65 0.60 0.55 0.50 0.45 0.40; do
        model_name="subj0${subj}_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_snr_${snr}"

            python Train.py \
            --data_path=../dataset \
            --cache_dir=../cache \
            --model_name=${model_name} \
            --no-multi_subject \
            --subj=${subj} \
            --dual_guidance \
            --caption_type="short" \
            --snr_threshold $snr

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
                --snr_threshold $snr
                
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

for subj in 4 6 8; do
    for snr in 0.70 0.65 0.60 0.55 0.50 0.45 0.40; do
        model_name="subj0${subj}_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_snr_${snr}"

            python Train.py \
            --data_path=../dataset \
            --cache_dir=../cache \
            --model_name=${model_name} \
            --no-multi_subject \
            --subj=${subj} \
            --dual_guidance \
            --caption_type="short" \
            --snr_threshold $snr

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
                --snr_threshold $snr
                
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