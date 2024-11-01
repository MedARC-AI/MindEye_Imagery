jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="2"
subj=1
for caption_type in "medium"; do
    model_name="subj0${subj}_40sess_hypatia_mirage_schmedium_train_medium_normalize"

    python Train.py \
        --data_path=../dataset \
        --cache_dir=../cache \
        --model_name=${model_name} \
        --no-multi_subject \
        --subj=${subj} \
        --weight_decay=100000 \
        --dual_guidance \
        --caption_type="schmedium"

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
            --caption_type="medium"
            
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

