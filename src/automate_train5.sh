jupyter nbconvert Train_ridge_dp.ipynb --to python
jupyter nbconvert recon_inference_mi_ridge_dp.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=32 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export CUDA_VISIBLE_DEVICES="3"

subj=1 

model_name="subj0${subj}_40sess_hypatia_ridge_sdxl_ip_adapter_plus_dp"
echo model_name=${model_name}

python Train_ridge_dp.py \
    --data_path=../dataset \
    --cache_dir=../cache \
    --model_name=${model_name} \
    --no-multi_subject \
    --max_lr=3e-5 \
    --num_sessions=40 \
    --use_prior \
    --subj=${subj} \
    --batch_size=${BATCH_SIZE}\
    --plus

for mode in "imagery" "vision"; do
        python recon_inference_mi_ridge_dp.py \
            --model_name $model_name \
            --subj $subj \
            --mode $mode \
            --cache_dir ../cache \
            --data_path ../dataset \
            --use_prior \
            --save_raw \
            --plus

    done