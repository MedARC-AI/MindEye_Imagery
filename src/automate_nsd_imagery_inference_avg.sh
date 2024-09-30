jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert recon_inference_mi_avg.ipynb --to python
jupyter nbconvert final_evaluations_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python
jupyter nbconvert Train.ipynb --to python
export CUDA_VISIBLE_DEVICES="0"

for subj in 1; do
    model_name="pretrained_subj01_40sess_hypatia_vd_dual_proj" 
    for mode in "vision" "imagery"; do # "vision" "imagery"; do

        # python recon_inference_mi_avg.py \
        #     --model_name $model_name \
        #     --subj $subj \
        #     --mode $mode \
        #     --cache_dir ../cache \
        #     --data_path ../dataset \
        #     --hidden_dim 1024 \
        #     --n_blocks 4 \
        #     --dual_guidance

        python final_evaluations_mi_multi.py \
            --model_name "${model_name}_avg" \
            --all_recons_path evals/${model_name}_avg/${model_name}_avg_all_recons_${mode}.pt \
            --subj $subj \
            --mode $mode \
            --data_path ../dataset \
            --cache_dir ../cache

        python final_evaluations_mi_multi.py \
            --model_name "${model_name}_avg2" \
            --all_recons_path evals/${model_name}_avg2/${model_name}_avg2_all_recons_${mode}.pt \
            --subj $subj \
            --mode $mode \
            --data_path ../dataset \
            --cache_dir ../cache
            # --no-blurry_recon

        python plots_across_subjects.py \
                --model_name="${model_name}_avg2" \
                --mode="${mode}" \
                --data_path ../dataset \
                --cache_dir ../cache \
                --criteria all \
                --all_recons_path evals/${model_name}_avg2/${model_name}_avg2_all_recons_${mode}.pt \
                --subjs=$subj
    done


python plots_across_methods.py \
--methods "mindeye1_subj01, \
braindiffuser_subj01, \
final_subj01_pretrained_40sess_24bs, \
pretrained_subj01_40sess_hypatia_vd2, \
pretrained_subj01_40sess_hypatia_vd_dual_proj, \
pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
pretrained_subj01_40sess_hypatia_vd_dual_proj_avg2" \
--data_path ../dataset \
--output_path ../figs/ \
--output_file methods_scatter_reduced.png

done

