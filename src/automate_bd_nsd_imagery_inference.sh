jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python
export CUDA_VISIBLE_DEVICES="0"

subj=1
for model_name in "braindiffuser_subj01_0.0_mix" "braindiffuser_subj01_0.4_mix" "braindiffuser_subj01_1.0_mix" ; do
    for mode in "vision" "imagery"; do # "vision" "imagery"; do

        python final_evaluations_mi_multi.py \
            --model_name "${model_name}" \
            --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
            --subj $subj \
            --mode $mode \
            --data_path ../dataset \
            --cache_dir ../cache
            # --no-blurry_recon

        python plots_across_subjects.py \
                --model_name="${model_name}" \
                --mode="${mode}" \
                --data_path ../dataset \
                --cache_dir ../cache \
                --criteria all \
                --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
                --subjs=$subj
    done

python plots_across_methods.py \
--methods "mindeye1_subj01, \
braindiffuser_subj01, \
final_subj01_pretrained_40sess_24bs, \
braindiffuser_subj01_0.0_mix, \
braindiffuser_subj01_0.4_mix, \
braindiffuser_subj01_1.0_mix" \
--data_path ../dataset \
--output_path ../figs/ \
--output_file methods_braindiffuser_mix.png


done

