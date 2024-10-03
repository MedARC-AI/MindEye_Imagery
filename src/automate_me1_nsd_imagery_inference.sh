jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert recon_inference_mi_avg.ipynb --to python
jupyter nbconvert final_evaluations_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python
jupyter nbconvert Train.ipynb --to python
export CUDA_VISIBLE_DEVICES="3"

subj=1
for model_name in "mindeye1_prior_257_final_subj01_bimixco_softclip_byol_replication"; do
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

done
python plots_across_methods.py \
--methods "mindeye1_subj01, \
final_subj01_pretrained_40sess_24bs, \
pretrained_subj01_40sess_hypatia_vd2, \
pretrained_subj01_40sess_hypatia_vd_dual_proj, \
mindeye1_subj01_hypatia_dual_proj2, \
prior_257_final_subj01_bimixco_softclip_byol_img2img0.85_16, \
braindiffuser_subj01, \
pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
mindeye1_subj01_hypatia_default, \
mindeye1_subj01_hypatia_default2, \
mindeye1_prior_257_final_subj01_bimixco_softclip_byol,
mindeye1_prior_257_final_subj01_bimixco_softclip_byol_replication" \
--data_path ../dataset \
--output_path ../figs/ \
--output_file "mindeye1_comparison"



