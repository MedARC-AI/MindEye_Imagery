jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert enhanced_recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
export CUDA_VISIBLE_DEVICES="1"

for subj in 2 5 7; do
    model_name="final_subj0${subj}_pretrained_40sess_24bs"
    for mode in "vision" "imagery"; do # "vision" "imagery"; do
        python recon_inference_mi.py \
            --model_name $model_name \
            --subj $subj \
            --mode $mode \
            --cache_dir ../cache \
            --data_path ../dataset \
            --hidden_dim 4096 \
            --n_blocks 4 \
            --gen_rep 10
            # --no-blurry_recon

        python enhanced_recon_inference_mi.py \
            --model_name $model_name \
            --subj $subj \
            --mode $mode
            # --no-blurry_recon

        # python final_evaluations_mi_multi.py \
        #     --model_name $model_name \
        #     --all_recons_path evals/${model_name}/${model_name}_all_enhancedrecons_${mode}.pt \
        #     --subj $subj \
        #     --mode $mode \
        #     --data_path ../dataset \
        #     --cache_dir ../cache
            # --no-blurry_recon
    done
done
# done
