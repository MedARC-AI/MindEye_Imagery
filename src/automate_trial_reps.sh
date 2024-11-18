jupyter nbconvert Train.ipynb --to python
jupyter nbconvert recon_inference_mi.ipynb --to python
jupyter nbconvert final_evaluations_mi_multi.ipynb --to python
jupyter nbconvert plots_across_subjects.ipynb --to python
jupyter nbconvert plots_across_methods.ipynb --to python

export CUDA_VISIBLE_DEVICES="1"
for subj in 1 2 5 7; do
        for trial_reps in {1..8}; do
                model_name="subj0${subj}_40sess_hypatia_mirage_${trial_reps}_trial_reps"

                mode="imagery"
                python recon_inference_mi.py \
                        --model_name $model_name \
                        --subj $subj \
                        --mode $mode \
                        --cache_dir ../cache \
                        --data_path ../dataset \
                        --save_raw \
                        --raw_path /export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/ \
                        --no-prompt_recon \
                        --num_trial_reps $trial_reps \
                        --no-retrieval
                        
                python final_evaluations_mi_multi.py \
                        --model_name $model_name \
                        --all_recons_path evals/${model_name}/${model_name}_all_recons_${mode}.pt \
                        --subj $subj \
                        --mode $mode \
                        --data_path ../dataset \
                        --cache_dir ../cache

                done
        done