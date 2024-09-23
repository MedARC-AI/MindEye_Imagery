cd /weka/proj-fmri/ckadirt/MindEye_Imagery/src
source /admin/home-ckadirt/mindeye/bin/activate


export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=21 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

subj=(1 2 5 7)  # Changed to array syntax for iteration

# multisubject inference
for s in "${subj[@]}"; do  # Iterate over subjects
    model_name="pretrained_subj0${s}_40sess_hypatia_vd2"  # Updated model name to include subject
    echo new_model_name=${model_name}

    jupyter nbconvert final_evaluations_mi_multi.ipynb --to python

    for mode in "imagery" "vision"; do
        all_recons_path="evals/${model_name}/${model_name}_all_recons_${mode}.pt"  # Corrected path assignment syntax
        python final_evaluations_mi_multi.py \
            --model_name=${model_name} \
            --subj=${s} \
            --mode=${mode} \
            --cache_dir=/weka/proj-medarc/shared/cache \
            --data_path=/weka/proj-medarc/shared/mindeyev2_dataset \
            --blurry_recon \
            --imagery_data_path=/weka/proj-medarc/shared/umn-imagery \
            --criteria=all \
            --all_recons_path=${all_recons_path} 
    done
done
