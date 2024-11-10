jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="0"

# singlesubject finetuning
model_name="pretrained_subj0${subj}_finetuning_vd_outputs"
data_path="/viscam/u/shubh/neuro/MindEye_Imagery/umn-imagery"

for subj in 1; do
    model_name="subj0${subj}_40sess_hypatia_mirage"

    # python Train.py \
    #     --data_path=${data_path} \
    #     --cache_dir=../cache \
    #     --model_name=${model_name} \
    #     --subj=${subj} \
    #     # --prompt_recon=False 

    for mode in "vision" "imagery" "shared1000"; do #

        python recon_inference_mi.py \
            --model_name $model_name \
            --subj $subj \
            --mode $mode \
            --cache_dir ../cache \
            --data_path=${data_path} \
            --save_raw \
            --raw_path /viscam/u/shubh/neuro/MindEye_Imagery/src/raw_outputs
            
        python final_evaluations_mi_multi.py \
                --model_name $model_name \
                --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
                --subj $subj \
                --mode $mode \
                --data_path=${data_path} \
                --cache_dir ../cache

        python plots_across_subjects.py \
                --model_name="${model_name}" \
                --mode="${mode}" \
                --data_path=${data_path} \
                --cache_dir ../cache \
                --criteria all \
                --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
                --subjs=$subj

        done
    done
