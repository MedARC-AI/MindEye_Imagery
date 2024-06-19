jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert enhanced_recon_inference_mi.ipynb --to python
export CUDA_VISIBLE_DEVICES="2"

for subj in 10; do
    # for gen_rep in 4 5 6 7 8 9; do
    for model in pretrained_subj${subj}irf_40sess_hypatia_imageryrf_all_no_blurry pretrained_subj${subj}irf_40sess_hypatia_imageryrf_vision_no_blurry; do
        for mode in "imagery" "vision"; do
            python recon_inference_mi.py \
                --model_name $model \
                --subj $subj \
                --mode $mode \
                --cache_dir ../cache \
                --data_path ../dataset \
                --hidden_dim 1024 \
                --n_blocks 4 \
                --no-blurry_recon

            python enhanced_recon_inference_mi.py \
                --model_name $model \
                --subj $subj \
                --mode $mode \
                --no-blurry_recon
         done
    done
done
