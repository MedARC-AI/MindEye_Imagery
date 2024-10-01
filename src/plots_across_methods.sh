jupyter nbconvert plots_across_methods.ipynb --to python

python plots_across_methods.py \
--methods "mindeye1_subj01, \
braindiffuser_subj01, \
final_subj01_pretrained_40sess_24bs, \
pretrained_subj01_40sess_hypatia_vd2, \
pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
pretrained_subj01_40sess_hypatia_vd_dual_proj_wd_60000" \
--data_path ../dataset \
--output_path ../figs/ \
--output_file methods_scatter_reduced.png



# python plots_across_methods.py \
# --methods "mindeye1_subj01, \
# braindiffuser_subj01, \
# final_subj01_pretrained_40sess_24bs, \
# pretrained_subj01_40sess_hypatia_vd2, \
# pretrained_subj01_40sess_hypatia_vd_snr_0_5, \
# pretrained_subj01_40sess_hypatia_vd_multisubject_snr_0_5, \
# pretrained_subj01_40sess_hypatia_vd_snr_0_55, \
# pretrained_subj01_40sess_hypatia_vd_multisubject_snr_0_6, \
# pretrained_subj01_40sess_hypatia_vd_snr_0_65, \
# pretrained_subj01_40sess_hypatia_vd_snr_0_75" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_snr

# python plots_across_methods.py \
# --methods "mindeye1_subj01, \
# braindiffuser_subj01, \
# final_subj01_pretrained_40sess_24bs, \
# pretrained_subj01_40sess_hypatia_vd2, \
# pretrained_subj01_40sess_hypatia_vd_dual_proj, \
# pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
# pretrained_subj01_40sess_hypatia_ip_adapter2, \
# pretrained_subj01_40sess_hypatia_ip_adapter_plus" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_snr


# python plots_across_methods.py \
# --methods "braindiffuser_subj01, \
# final_subj01_pretrained_40sess_24bs, \
# subj01_40sess_hypatia_turbo_ridge_alpha_0.01, \
# subj01_40sess_hypatia_turbo_ridge_alpha_0.1, \
# subj01_40sess_hypatia_turbo_ridge_alpha_1, \
# subj01_40sess_hypatia_turbo_ridge_alpha_10, \
# subj01_40sess_hypatia_turbo_ridge_alpha_100, \
# subj01_40sess_hypatia_turbo_ridge_alpha_1000, \
# subj01_40sess_hypatia_turbo_ridge_alpha_10000, \
# subj01_40sess_hypatia_turbo_ridge_alpha_60000, \
# subj01_40sess_hypatia_turbo_ridge_alpha_100000" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_ridge_alpha