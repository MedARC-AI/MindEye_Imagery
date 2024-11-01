jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi_final.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="1"
for subj in 2; do
    # model_name="subj0${subj}_40sess_hypatia_mirage_no_retrieval"

    # python Train.py \
    #     --data_path=../dataset \
    #     --cache_dir=../cache \
    #     --model_name=${model_name} \
    #     --subj=${subj} \
    #     --no-retrieval

    # for mode in "vision" "imagery"; do # "shared1000"

    #     python recon_inference_mi.py \
    #         --model_name $model_name \
    #         --subj $subj \
    #         --mode $mode \
    #         --cache_dir ../cache \
    #         --data_path ../dataset \
    #         --save_raw \
    #         --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
    #         --no-retrieval \
    #         --no-compile_models
            
    #     python final_evaluations_mi_multi.py \
    #             --model_name $model_name \
    #             --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
    #             --subj $subj \
    #             --mode $mode \
    #             --data_path ../dataset \
    #             --cache_dir ../cache

    #     python plots_across_subjects.py \
    #             --model_name="${model_name}" \
    #             --mode="${mode}" \
    #             --data_path ../dataset \
    #             --cache_dir ../cache \
    #             --criteria all \
    #             --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
    #             --subjs=$subj

    # done

    model_name="subj0${subj}_40sess_hypatia_mirage2"

    python Train.py \
        --data_path=../dataset \
        --cache_dir=../cache \
        --model_name=${model_name} \
        --subj=${subj} 

    for mode in "vision" "imagery"; do # "shared1000"

        python recon_inference_mi_final.py \
            --model_name $model_name \
            --subj $subj \
            --mode $mode \
            --cache_dir ../cache \
            --data_path ../dataset \
            --save_raw \
            --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ 
            
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