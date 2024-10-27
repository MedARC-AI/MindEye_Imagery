jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="0"

subj=1 
# model_name="subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae_sharpness"
# echo model_name=${model_name}

# # python Train.py \
# #             --data_path=../dataset \
# #             --cache_dir=../cache \
# #             --model_name=${model_name} \
# #             --no-multi_subject \
# #             --subj=${subj} \
# #             --dual_guidance

# for mode in "vision" "imagery"; do #

#     python recon_inference_mi.py \
#         --model_name $model_name \
#         --subj $subj \
#         --mode $mode \
#         --cache_dir ../cache \
#         --data_path ../dataset \
#         --save_raw \
#         --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
#         --dual_guidance \
#         --vdvae \
#         --strength 0.70 \
#         --filter_sharpness
        
#     python final_evaluations_mi_multi.py \
#             --model_name $model_name \
#             --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#             --subj $subj \
#             --mode $mode \
#             --data_path ../dataset \
#             --cache_dir ../cache

#     python plots_across_subjects.py \
#             --model_name="${model_name}" \
#             --mode="${mode}" \
#             --data_path ../dataset \
#             --cache_dir ../cache \
#             --criteria all \
#             --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#             --subjs=$subj

#     done

model_name="subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae_sharpness_color"
echo model_name=${model_name}

# python Train.py \
#             --data_path=../dataset \
#             --cache_dir=../cache \
#             --model_name=${model_name} \
#             --no-multi_subject \
#             --subj=${subj} \
#             --dual_guidance

for mode in "vision" "imagery"; do #

    python recon_inference_mi.py \
        --model_name $model_name \
        --subj $subj \
        --mode $mode \
        --cache_dir ../cache \
        --data_path ../dataset \
        --save_raw \
        --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
        --dual_guidance \
        --vdvae \
        --strength 0.70 \
        --filter_sharpness \
        --filter_color
        
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

# model_name="subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae_contrast_sharpness"
# echo model_name=${model_name}

# python Train.py \
#             --data_path=../dataset \
#             --cache_dir=../cache \
#             --model_name=${model_name} \
#             --no-multi_subject \
#             --subj=${subj} \
#             --dual_guidance

# for mode in "vision" "imagery"; do #

#     python recon_inference_mi.py \
#         --model_name $model_name \
#         --subj $subj \
#         --mode $mode \
#         --cache_dir ../cache \
#         --data_path ../dataset \
#         --save_raw \
#         --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
#         --dual_guidance \
#         --strength 0.70 \
#         --vdvae \
#         --filter_contrast \
#         --filter_sharpness
        
#     python final_evaluations_mi_multi.py \
#             --model_name $model_name \
#             --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#             --subj $subj \
#             --mode $mode \
#             --data_path ../dataset \
#             --cache_dir ../cache

#     python plots_across_subjects.py \
#             --model_name="${model_name}" \
#             --mode="${mode}" \
#             --data_path ../dataset \
#             --cache_dir ../cache \
#             --criteria all \
#             --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#             --subjs=$subj

#     done

model_name="subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae_contrast_sharpness_color"
echo model_name=${model_name}

# python Train.py \
#             --data_path=../dataset \
#             --cache_dir=../cache \
#             --model_name=${model_name} \
#             --no-multi_subject \
#             --subj=${subj} \
#             --dual_guidance

for mode in "vision" "imagery"; do #

    python recon_inference_mi.py \
        --model_name $model_name \
        --subj $subj \
        --mode $mode \
        --cache_dir ../cache \
        --data_path ../dataset \
        --save_raw \
        --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
        --dual_guidance \
        --strength 0.70 \
        --vdvae \
        --filter_contrast \
        --filter_sharpness \
        --filter_color
        
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

model_name="subj01_40sess_hypatia_ridge_sc_0.70_strength_contrast"
echo model_name=${model_name}

python Train.py \
            --data_path=../dataset \
            --cache_dir=../cache \
            --model_name=${model_name} \
            --no-multi_subject \
            --subj=${subj} \
            --dual_guidance

for mode in "vision" "imagery"; do #

    python recon_inference_mi.py \
        --model_name $model_name \
        --subj $subj \
        --mode $mode \
        --cache_dir ../cache \
        --data_path ../dataset \
        --save_raw \
        --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
        --dual_guidance \
        --strength 0.70 \
        --filter_contrast
        
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

model_name="subj01_40sess_hypatia_ridge_sc_0.70_strength_contrast_sharpness"
echo model_name=${model_name}

python Train.py \
            --data_path=../dataset \
            --cache_dir=../cache \
            --model_name=${model_name} \
            --no-multi_subject \
            --subj=${subj} \
            --dual_guidance

for mode in "vision" "imagery"; do #

    python recon_inference_mi.py \
        --model_name $model_name \
        --subj $subj \
        --mode $mode \
        --cache_dir ../cache \
        --data_path ../dataset \
        --save_raw \
        --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
        --dual_guidance \
        --strength 0.70 \
        --filter_contrast \
        --filter_sharpness
        
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

model_name="subj01_40sess_hypatia_ridge_sc_0.70_strength_contrast_sharpness_color"
echo model_name=${model_name}

python Train.py \
            --data_path=../dataset \
            --cache_dir=../cache \
            --model_name=${model_name} \
            --no-multi_subject \
            --subj=${subj} \
            --dual_guidance

for mode in "vision" "imagery"; do #

    python recon_inference_mi.py \
        --model_name $model_name \
        --subj $subj \
        --mode $mode \
        --cache_dir ../cache \
        --data_path ../dataset \
        --save_raw \
        --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
        --dual_guidance \
        --strength 0.70 \
        --filter_contrast \
        --filter_sharpness \
        --filter_color
        
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

# for strength in 0.70 0.75 0.65 0.80; do # 0.70 0.80 0.65 
#     for subj in 2 5 7; do
#         # model_name="subj0${subj}_40sess_hypatia_ridge_sc_${strength}_strength_vdvae"
#         # echo model_name=${model_name}

#         # python Train.py \
#         #     --data_path=../dataset \
#         #     --cache_dir=../cache \
#         #     --model_name=${model_name} \
#         #     --no-multi_subject \
#         #     --subj=${subj} \
#         #     --dual_guidance \

#         # for mode in "vision" "imagery"; do #

#         #     python recon_inference_mi.py \
#         #         --model_name $model_name \
#         #         --subj $subj \
#         #         --mode $mode \
#         #         --cache_dir ../cache \
#         #         --data_path ../dataset \
#         #         --save_raw \
#         #         --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
#         #         --dual_guidance \
#         #         --strength $strength \
#         #         --vdvae
                
#         #     python final_evaluations_mi_multi.py \
#         #             --model_name $model_name \
#         #             --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#         #             --subj $subj \
#         #             --mode $mode \
#         #             --data_path ../dataset \
#         #             --cache_dir ../cache

#         #     python plots_across_subjects.py \
#         #             --model_name="${model_name}" \
#         #             --mode="${mode}" \
#         #             --data_path ../dataset \
#         #             --cache_dir ../cache \
#         #             --criteria all \
#         #             --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#         #             --subjs=$subj

#         #     done

#         model_name="subj0${subj}_40sess_hypatia_ridge_sc_${strength}_strength_filtered"
#         echo model_name=${model_name}

#         python Train.py \
#             --data_path=../dataset \
#             --cache_dir=../cache \
#             --model_name=${model_name} \
#             --no-multi_subject \
#             --subj=${subj} \
#             --dual_guidance \

#         for mode in "vision" "imagery"; do #

#             python recon_inference_mi.py \
#                 --model_name $model_name \
#                 --subj $subj \
#                 --mode $mode \
#                 --cache_dir ../cache \
#                 --data_path ../dataset \
#                 --save_raw \
#                 --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
#                 --dual_guidance \
#                 --strength $strength \
#                 --filter_blurry
                
#             python final_evaluations_mi_multi.py \
#                     --model_name $model_name \
#                     --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#                     --subj $subj \
#                     --mode $mode \
#                     --data_path ../dataset \
#                     --cache_dir ../cache

#             python plots_across_subjects.py \
#                     --model_name="${model_name}" \
#                     --mode="${mode}" \
#                     --data_path ../dataset \
#                     --cache_dir ../cache \
#                     --criteria all \
#                     --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
#                     --subjs=$subj

#             done
#         done
#     done

python plots_across_methods.py \
    --methods "mindeye1_subj01, \
    braindiffuser_subj01, \
    subj01_40sess_hypatia_ridge_sc3, \
    subj01_40sess_hypatia_ridge_sc_vdvae, \
    subj01_40sess_hypatia_ridge_sc_0.65_strength_vdvae, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae, \
    subj01_40sess_hypatia_ridge_sc_0.60_strength, \
    subj01_40sess_hypatia_ridge_sc_0.65_strength, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength, \
    subj01_40sess_hypatia_ridge_sc_0.80_strength, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae_sharpness, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae_contrast_sharpness, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae_sharpness_color, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae_contrast_sharpness_color, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength_contrast, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength_contrast_sharpness, \
    subj01_40sess_hypatia_ridge_sc_0.70_strength_contrast_sharpness_color" \
    --data_path ../dataset \
    --output_path ../figs/ \
    --output_file methods_scatter_sc_low_level

    #,subj01_40sess_hypatia_ridge_sc_medium_captions