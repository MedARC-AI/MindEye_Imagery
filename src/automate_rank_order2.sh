jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="2"
for subj in 6 8; do
    for num_rois in {2..43}; do 
        model_name="subj0${subj}_40sess_hypatia_ridge_scv_nsd_general_RO_rois_${num_rois}"
        echo model_name=${model_name}
        python Train.py \
            --data_path=../dataset \
            --cache_dir=../cache \
            --model_name=${model_name} \
            --no-multi_subject \
            --subj=${subj} \
            --dual_guidance \
            --caption_type="short" \
            --top_n_rank_order_rois ${num_rois} \
            --nsd_general_rank_order_rois

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
                --top_n_rank_order_rois ${num_rois} \
                --samplewise_rank_order_rois
                
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