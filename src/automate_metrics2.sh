jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="2"
# for subj in 1; do
#     for model_name in "subj0${subj}_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_schmedium_captions"; do #"subj0${subj}_40sess_hypatia_mirage_no_filters" "subj0${subj}_40sess_hypatia_mirage_no_blurry" "subj0${subj}_40sess_hypatia_mirage_no_retrieval"


#         for mode in "vision" "imagery"; do # "shared1000"
                
#             python final_evaluations_mi_multi.py \
#                     --model_name $model_name \
#                     --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#                     --subj $subj \
#                     --mode $mode \
#                     --data_path ../dataset \
#                     --cache_dir ../cache 

#             # python plots_across_subjects.py \
#             #         --model_name="${model_name}" \
#             #         --mode="${mode}" \
#             #         --data_path ../dataset \
#             #         --cache_dir ../cache \
#             #         --criteria all \
#             #         --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#             #         --subjs=$subj

#             done
#         done
#     done

for subj in 1 2 5 7; do
    for model_name in "final_subj0${subj}_pretrained_40sess_24bs"; do # "mindeye1_subj0${subj}""mindeye1_subj0${subj}" "braindiffuser_subj0${subj}" "subj0${subj}_40sess_hypatia_mirage2" "subj0${subj}_40sess_hypatia_mirage5"; do


        for mode in "vision" "imagery"; do # "shared1000"
                
            python final_evaluations_mi_multi.py \
                    --model_name $model_name \
                    --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
                    --subj $subj \
                    --mode $mode \
                    --data_path ../dataset \
                    --cache_dir ../cache 

            # python plots_across_subjects.py \
            #         --model_name="${model_name}" \
            #         --mode="${mode}" \
            #         --data_path ../dataset \
            #         --cache_dir ../cache \
            #         --criteria all \
            #         --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
            #         --subjs=$subj

            done
        done
    done