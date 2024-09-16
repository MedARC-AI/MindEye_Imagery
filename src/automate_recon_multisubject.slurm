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

# Make sure you activate your fmri environment created from src/setup.sh
cd /weka/proj-fmri/ckadirt/MindEye_Imagery/src
source /admin/home-ckadirt/mindeye/bin/activate

# The following line converts your jupyter notebook into a python script runnable with Slurm
jupyter nbconvert Train.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=21 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

subj=(1 2 5 7)  # Changed to array syntax for iteration

# multisubject inference
for s in "${subj[@]}"; do  # Iterate over subjects
    model_name="pretrained_subj0${s}_40sess_hypatia_vd2"  # Updated model name to include subject
    echo new_model_name=${model_name}

    jupyter nbconvert recon_inference_mi_vd.ipynb --to python

    for mode in "imagery" "vision"; do
        python recon_inference_mi_vd.py \
            --model_name=${model_name} \
            --subj=${s} \
            --mode=${mode} \
            --cache_dir=/weka/proj-medarc/shared/cache \
            --data_path=/weka/proj-medarc/shared/umn-imagery \
            --hidden_dim=1024 \
            --n_blocks=4 \
            --blurry_recon \
            --imagery_data_path=/weka/proj-medarc/shared/umn-imagery
    done
done
