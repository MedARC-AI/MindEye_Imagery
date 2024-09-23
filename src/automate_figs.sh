cd /weka/proj-fmri/ckadirt/MindEye_Imagery/src
source /admin/home-ckadirt/mindeye/bin/activate

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=21 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

model_name="pretrained_subj01_40sess_hypatia_vd2"  # Updated model name to include subject
echo new_model_name=${model_name}

jupyter nbconvert plots_across_subjects.ipynb --to python

for mode in "imagery" "vision"; do
    all_recons_path="evals/${model_name}/${model_name}_all_recons_${mode}.pt"  # Corrected path assignment syntax
    for criteria in "SSIM" "SwAV" "EffNet-B" "PixCorr" "Brain*Corr.*nsd_general" "all"; do
        python plots_across_subjects.py \
            --model_name=${model_name} \
            --mode=${mode} \
            --cache_dir=/weka/proj-medarc/shared/cache \
            --data_path=/weka/proj-medarc/shared/mindeyev2_dataset \
            --imagery_data_path=/weka/proj-medarc/shared/umn-imagery \
            --criteria=${criteria} \
            --all_recons_path=${all_recons_path}
    done
done
