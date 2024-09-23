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

# Load necessary modules or environments if required
# Example: module load python/3.8

# Navigate to the project directory
cd /weka/proj-fmri/ckadirt/MindEye_Imagery/src

# Activate the Python environment
source /admin/home-ckadirt/mindeye/bin/activate

# Convert the Jupyter notebooks to Python scripts
jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi_vd.ipynb --to python

# Set environment variables
export NUM_GPUS=1  # Number of GPUs to use
export BATCH_SIZE=21 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Define the session numbers for subject 1
sessions=(1 2 5 10 20)

# Define the subject (only subject 1 for the new models)
subj=1

# Iterate over each session number
for session in "${sessions[@]}"; do
    # Construct the model name based on the current session
    model_name="pretrained_subj0${subj}_40sess_hypatia_vd2_sessions${session}"
    echo "Running inference with model: ${model_name}"

    # Optional: Re-convert the inference notebook if there are changes
    # jupyter nbconvert recon_inference_mi_vd.ipynb --to python

    # Iterate over each mode
    for mode in "imagery" "vision"; do
        echo "Mode: ${mode}"
        
        # Execute the Python inference script with the specified parameters
        python recon_inference_mi_vd.py \
            --model_name="${model_name}" \
            --subj="${subj}" \
            --mode="${mode}" \
            --cache_dir="/weka/proj-medarc/shared/cache" \
            --data_path="/weka/proj-medarc/shared/umn-imagery" \
            --hidden_dim=1024 \
            --n_blocks=4 \
            --blurry_recon \
            --imagery_data_path="/weka/proj-medarc/shared/umn-imagery"
        
        # Check if the Python script executed successfully
        if [ $? -ne 0 ]; then
            echo "Inference failed for model ${model_name} in mode ${mode}"
            exit 1
        fi
    done
done

echo "All inferences completed successfully."
