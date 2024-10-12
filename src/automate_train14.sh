
jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=32 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export CUDA_VISIBLE_DEVICES="3"

subj=1 

model_name="subj0${subj}_40sess_hypatia_ridge_flat_dp5"
echo model_name=${model_name}

python Train.py \
    --data_path=../dataset \
    --cache_dir=../cache \
    --model_name=${model_name} \
    --no-multi_subject \
    --max_lr=3e-5 \
    --num_sessions=40 \
    --use_prior \
    --subj=${subj} \
    --batch_size=${BATCH_SIZE} \
    --dual_guidance 