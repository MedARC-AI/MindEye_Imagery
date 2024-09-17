#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=p5
#SBATCH --job-name=plots_across_subjects
#SBATCH --nodes=1           
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --error=slurms/%j.err    # first create a "slurms" folder in current directory to store logs
#SBATCH --output=slurms/%j.out
#SBATCH --comment=medarc
#SBATCH --no-requeue
#SBATCH --qos=idle

# Load the necessary environment
cd /weka/proj-fmri/ckadirt/MindEye_Imagery/src
source /admin/home-ckadirt/mindeye/bin/activate

# Set environment variables
export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=21 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Session numbers array for subject 1
sessions=(1 2 5 10 20)

# Convert Jupyter notebook to Python script
jupyter nbconvert plots_across_subjects.ipynb --to python

# Loop over session numbers
for session in "${sessions[@]}"; do
    # Construct model name for the current session
    model_name="pretrained_subj01_40sess_hypatia_vd2_sessions${session}"
    echo "Processing model: ${model_name}"

    # Loop over modes: imagery and vision
    for mode in "imagery" "vision"; do
        # Set path for all reconstructions based on mode
        all_recons_path="evals/${model_name}/${model_name}_all_recons_${mode}.pt"
        
        # Loop over criteria
        for criteria in "SSIM" "SwAV" "EffNet-B" "PixCorr" "Brain*Corr.*nsd_general" "all"; do
            echo "Running plots for criteria: ${criteria} in mode: ${mode}"

            # Run Python script for the given criteria and mode
            python plots_across_subjects.py \
                --model_name="${model_name}" \
                --mode="${mode}" \
                --cache_dir="/weka/proj-medarc/shared/cache" \
                --data_path="/weka/proj-medarc/shared/mindeyev2_dataset" \
                --imagery_data_path="/weka/proj-medarc/shared/umn-imagery" \
                --criteria="${criteria}" \
                --all_recons_path="${all_recons_path}" \
                --subjs="1"
            
            # Check if the Python script was successful
            if [ $? -ne 0 ]; then
                echo "Plot generation failed for model ${model_name}, mode ${mode}, criteria ${criteria}"
                exit 1
            fi
        done
    done
done

echo "All plotting tasks completed successfully."
