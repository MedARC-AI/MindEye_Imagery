#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=p5
#SBATCH --job-name=recon_inference_mi_vd
#SBATCH --nodes=1           
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=slurms/%j.err    # first create a "slurms" folder in current directory to store logs
#SBATCH --output=slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue
#SBATCH --qos=idle

# Navigate to the project directory and activate the environment
cd /weka/proj-fmri/ckadirt/MindEye_Imagery/src
source /admin/home-ckadirt/mindeye/bin/activate

# Set environment variables
export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=21 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Define session numbers for subject 1
sessions=(1 2 5 10 20)

# Convert Jupyter notebook to Python script
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python

# Iterate over session numbers for subject 1
subj=1  # Fixed subject 1
for session in "${sessions[@]}"; do
    # Construct model name based on the session
    model_name="pretrained_subj0${subj}_40sess_hypatia_vd2_sessions${session}"
    echo "Processing model: ${model_name}"

    # Iterate over modes: imagery and vision
    for mode in "imagery" "vision"; do
        all_recons_path="evals/${model_name}/${model_name}_all_recons_${mode}.pt"  # Path for all reconstructions
        
        # Run the final evaluation script with the updated parameters
        python final_evaluations_mi_multi.py \
            --model_name=${model_name} \
            --subj=${subj} \
            --mode=${mode} \
            --cache_dir=/weka/proj-medarc/shared/cache \
            --data_path=/weka/proj-medarc/shared/mindeyev2_dataset \
            --blurry_recon \
            --imagery_data_path=/weka/proj-medarc/shared/umn-imagery \
            --criteria=all \
            --all_recons_path=${all_recons_path}
        
        # Check if the Python script was successful
        if [ $? -ne 0 ]; then
            echo "Evaluation failed for model ${model_name} in mode ${mode}"
            exit 1
        fi
    done
done

echo "All evaluations completed successfully."
