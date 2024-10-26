jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi_sd35.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="1"

subj=1 

model_name="subj01_40sess_hypatia_ridge_sd35_no_blurry"
echo model_name=${model_name}

# python Train.py \
#     --data_path=../dataset \
#     --cache_dir=../cache \
#     --model_name=${model_name} \
#     --no-multi_subject \
#     --subj=${subj} \
#     --weight_decay=60000

for mode in "imagery"; do #"vision" 

    python recon_inference_mi_sd35.py \
        --model_name $model_name \
        --subj $subj \
        --mode $mode \
        --cache_dir ../cache \
        --data_path ../dataset \
        --save_raw \
        --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
        --no-blurry_recon
        
    python final_evaluations_mi_multi.py \
            --model_name $model_name \
            --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
            --subj $subj \
            --mode $mode \
            --data_path ../dataset \
            --cache_dir ../cache

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
    pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
    subj01_40sess_hypatia_turbo_ridge_flat3, \
    subj01_40sess_hypatia_ridge_sc3,
    subj01_40sess_hypatia_ridge_sc_flux,
    subj01_40sess_hypatia_ridge_sc_flux_enhanced,
    subj01_40sess_hypatia_ridge_sc_sd35,
    subj01_40sess_hypatia_ridge_sc_sd35_enhanced,
    subj01_40sess_hypatia_ridge_sd35" \
    --data_path ../dataset \
    --output_path ../figs/ \
    --output_file methods_scatter_reduced4