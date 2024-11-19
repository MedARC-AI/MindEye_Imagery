jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="0"

# for subj in 1; do
#     for model_name in "subj0${subj}_40sess_hypatia_mirage_ts0"; do #"subj0${subj}_40sess_hypatia_mirage_no_filters" "subj0${subj}_40sess_hypatia_mirage_no_blurry" "subj0${subj}_40sess_hypatia_mirage_no_retrieval"


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

# for subj in 1 2 5 7; do
#     for model_name in "takagi_subj0${subj}"; do #"final_subj0${subj}_pretrained_40sess_24bs" "mindeye1_subj0${subj}" "braindiffuser_subj0${subj}" "subj0${subj}_40sess_hypatia_mirage2" "subj0${subj}_40sess_hypatia_mirage5"; do


#         for mode in "vision" "imagery" "shared1000"; do # 
                
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

# for subj in 1; do
#     for sess in 0.5 1 2 3 5 10 20 40; do
#        for model_name in "braindiffuser_subj0${subj}_${sess}sess"; do #"subj0${subj}_${sess}sess_hypatia_mirage"


#             for mode in "vision" "imagery" ; do # "shared1000"
                    
#                 python final_evaluations_mi_multi.py \
#                         --model_name $model_name \
#                         --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#                         --subj $subj \
#                         --mode $mode \
#                         --data_path ../dataset \
#                         --cache_dir ../cache

#                 # python plots_across_subjects.py \
#                 #         --model_name="${model_name}" \
#                 #         --mode="${mode}" \
#                 #         --data_path ../dataset \
#                 #         --cache_dir ../cache \
#                 #         --criteria all \
#                 #         --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#                 #         --subjs=$subj

#                 done
#             done
#         done
#     done

for subj in 1 2 5 7; do
    mode="imagery"
    for trial_reps in {1..16}; do
        for model_name in "mindeye_subj0${subj}_${trial_reps}_trial_reps" "braindiffuser_subj0${subj}_${trial_reps}_trial_reps" "mindeye2_subj0${subj}_${trial_reps}_trial_reps"; do
            python final_evaluations_mi_multi.py \
                    --model_name $model_name \
                    --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
                    --subj $subj \
                    --mode $mode \
                    --data_path ../dataset \
                    --cache_dir ../cache 

        done
    done

    mode="vision"
    for trial_reps in {1..8}; do
        for model_name in "mindeye_subj0${subj}_${trial_reps}_trial_reps" "braindiffuser_subj0${subj}_${trial_reps}_trial_reps" "mindeye2_subj0${subj}_${trial_reps}_trial_reps"; do
            python final_evaluations_mi_multi.py \
                    --model_name $model_name \
                    --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
                    --subj $subj \
                    --mode $mode \
                    --data_path ../dataset \
                    --cache_dir ../cache 
        done
    done
done