jupyter nbconvert plots_across_methods.ipynb --to python

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     pretrained_subj01_40sess_hypatia_vd2, \
#     pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
#     subj01_40sess_hypatia_turbo_ridge_flat3, \
#     subj01_40sess_hypatia_ridge_sc3, \
#     subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30"  \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "Brain Corr. nsd_general,Brain Corr. V1,Brain Corr. V2,Brain Corr. V3,Brain Corr. V4,Brain Corr. higher_vis" \
#     --output_file methods_scatter_reduced_5_brain_correlation
#     # --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5)" \
#     # --criteria "InceptionV3,CLIP,EffNet-B,SwAV" \

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     pretrained_subj01_40sess_hypatia_vd2, \
#     pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
#     subj01_40sess_hypatia_turbo_ridge_flat3, \
#     subj01_40sess_hypatia_ridge_sc3, \
#     subj01_40sess_hypatia_ridge_sc_vdvae, \
#     subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30"  \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --stimtype all \
#     --output_file methods_scatter_reduced_5

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     pretrained_subj01_40sess_hypatia_vd2, \
#     pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
#     subj01_40sess_hypatia_turbo_ridge_flat3, \
#     subj01_40sess_hypatia_ridge_sc3, \
#     subj01_40sess_hypatia_ridge_sc_vdvae, \
#     subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30"  \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --stimtype complex \
#     --output_file methods_scatter_reduced_5_complex

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     pretrained_subj01_40sess_hypatia_vd2, \
#     pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
#     subj01_40sess_hypatia_turbo_ridge_flat3, \
#     subj01_40sess_hypatia_ridge_sc3, \
#     subj01_40sess_hypatia_ridge_sc_vdvae, \
#     subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30"  \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --stimtype simple \
#     --output_file methods_scatter_reduced_5_simple

#     python plots_across_methods.py \
# --methods "mindeye1_subj01, \
# braindiffuser_subj01, \
# final_subj01_pretrained_40sess_24bs, \
# pretrained_subj01_40sess_hypatia_vd2, \
# pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
# subj01_40sess_hypatia_turbo_ridge_flat,
# subj01_40sess_hypatia_turbo_ridge_seq,
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_scatter_reduced.png




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

# python plots_across_methods.py \
# --methods "subj03_40sess_hypatia_nsd_general, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_1, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_2, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_3, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_4, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_5, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_6, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_7, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_8, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_9, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_10, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_11, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_12, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_13, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_14, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_15, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_16, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_17, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_18, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_19, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_20, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_21, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_22, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_23, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_24, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_25, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_26, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_27, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_28, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_29, \
# subj03_40sess_hypatia_ridge_rank_order_rois_samplewise_30" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s3

# python plots_across_methods.py \
# --methods "subj04_40sess_hypatia_nsd_general, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_1, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_2, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_3, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_4, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_5, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_6, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_7, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_8, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_9, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_10, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_11, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_12, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_13, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_14, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_15, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_16, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_17, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_18, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_19, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_20, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_21, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_22, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_23, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_24, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_25, \_ridge_
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_26, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_27, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_28, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_29, \
# subj04_40sess_hypatia_ridge_rank_order_rois_samplewise_30" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s4

# python plots_across_methods.py \
# --methods "subj06_40sess_hypatia_nsd_general, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_1, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_2, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_3, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_4, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_5, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_6, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_7, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_8, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_9, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_10, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_11, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_12, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_13, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_14, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_15, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_16, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_17, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_18, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_19, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_20, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_21, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_22, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_23, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_24, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_25, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_26, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_27, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_28, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_29, \
# subj06_40sess_hypatia_ridge_rank_order_rois_samplewise_30" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s6

# python plots_across_methods.py \
# --methods "subj08_40sess_hypatia_nsd_general, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_1, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_2, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_3, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_4, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_5, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_6, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_7, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_8, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_9, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_10, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_11, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_12, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_13, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_14, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_15, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_16, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_17, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_18, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_19, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_20, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_21, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_22, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_23, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_24, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_25, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_26, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_27, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_28, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_29, \
# subj08_40sess_hypatia_ridge_rank_order_rois_samplewise_30" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s8

# python plots_across_methods.py \
# --methods "subj03_40sess_hypatia_sc_nsd_general, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_1, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_2, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_3, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_4, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_5, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_6, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_7, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_8, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_9, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_10, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_11, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_12, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_13, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_14, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_15, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_16, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_17, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_18, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_19, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_20, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_21, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_22, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_23, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_25, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_26, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_27, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_28, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_29, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_31, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_32, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_33, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_34, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_35, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_36, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_37, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_38, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_39, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_40" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s3_samplewise_sc \
# --gradient


# python plots_across_methods.py \
# --methods "subj06_40sess_hypatia_sc_nsd_general, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_1, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_2, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_3, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_4, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_5, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_6, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_7, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_8, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_9, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_10, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_11, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_12, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_13, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_14, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_15, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_16, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_17, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_18, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_19, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_20, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_21, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_22, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_23, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_24, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_25, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_26, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_27, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_28, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_29, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_31, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_32, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_33, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_34, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_35, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_36, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_37, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_38, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_39, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_40" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s6_samplewise_sc \
# --gradient

# # mindeye1_subj01, \
# # final_subj01_pretrained_40sess_24bs, \

# python plots_across_methods.py \
# --methods "braindiffuser_subj01, \
# subj01_40sess_hypatia_sc_nsd_general, \
# subj01_40sess_hypatia_ridge_sc3, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_10, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_11, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_12, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_13, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_14, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_15, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_16, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_17, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_18, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_19, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_20, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_21, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_22, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_23, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_24, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_25, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_26, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_27, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_28, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_29, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_31, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_32, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_33, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_34, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_35, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_36, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_37, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_38, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_39, \
# subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_40" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s1_samplewise_sc \
# --gradient

# python plots_across_methods.py \
# --methods "subj03_40sess_hypatia_sc_nsd_general, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_1, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_2, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_3, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_4, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_5, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_6, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_7, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_8, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_9, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_10, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_11, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_12, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_13, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_14, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_15, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_16, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_17, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_18, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_19, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_20, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_21, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_22, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_23, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_25, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_26, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_27, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_28, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_29, \
# subj03_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_30" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s3_voxelwise_sc \
# --gradient


# python plots_across_methods.py \
# --methods "subj06_40sess_hypatia_sc_nsd_general, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_1, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_2, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_3, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_4, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_5, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_6, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_7, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_8, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_9, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_10, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_11, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_12, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_13, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_14, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_15, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_16, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_17, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_18, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_19, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_20, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_21, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_22, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_23, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_24, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_25, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_26, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_27, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_28, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_29, \
# subj06_40sess_hypatia_sc_ridge_rank_order_rois_voxelwise_30" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s6_voxelwise_sc \
# --gradient



# python plots_across_methods.py \
# --methods "braindiffuser_subj01, \
# subj01_40sess_hypatia_nsd_general, \
# subj01_40sess_hypatia_ridge_rank_order_rois_1, \
# subj01_40sess_hypatia_ridge_rank_order_rois_2, \
# subj01_40sess_hypatia_ridge_rank_order_rois_3, \
# subj01_40sess_hypatia_ridge_rank_order_rois_4, \
# subj01_40sess_hypatia_ridge_rank_order_rois_5, \
# subj01_40sess_hypatia_ridge_rank_order_rois_6, \
# subj01_40sess_hypatia_ridge_rank_order_rois_7, \
# subj01_40sess_hypatia_ridge_rank_order_rois_8, \
# subj01_40sess_hypatia_ridge_rank_order_rois_9, \
# subj01_40sess_hypatia_ridge_rank_order_rois_10, \
# subj01_40sess_hypatia_ridge_rank_order_rois_11, \
# subj01_40sess_hypatia_ridge_rank_order_rois_12, \
# subj01_40sess_hypatia_ridge_rank_order_rois_13, \
# subj01_40sess_hypatia_ridge_rank_order_rois_14, \
# subj01_40sess_hypatia_ridge_rank_order_rois_15, \
# subj01_40sess_hypatia_ridge_rank_order_rois_16, \
# subj01_40sess_hypatia_ridge_rank_order_rois_17, \
# subj01_40sess_hypatia_ridge_rank_order_rois_18, \
# subj01_40sess_hypatia_ridge_rank_order_rois_19, \
# subj01_40sess_hypatia_ridge_rank_order_rois_20, \
# subj01_40sess_hypatia_ridge_rank_order_rois_21, \
# subj01_40sess_hypatia_ridge_rank_order_rois_22, \
# subj01_40sess_hypatia_ridge_rank_order_rois_23, \
# subj01_40sess_hypatia_ridge_rank_order_rois_24, \
# subj01_40sess_hypatia_ridge_rank_order_rois_25, \
# subj01_40sess_hypatia_ridge_rank_order_rois_26, \
# subj01_40sess_hypatia_ridge_rank_order_rois_27, \
# subj01_40sess_hypatia_ridge_rank_order_rois_28, \
# subj01_40sess_hypatia_ridge_rank_order_rois_29, \
# subj01_40sess_hypatia_ridge_rank_order_rois_30" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s1_vd \
# --gradient

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     pretrained_subj01_40sess_hypatia_vd_dual_proj_avg, \
#     subj01_40sess_hypatia_turbo_ridge_flat3, \
#     subj01_40sess_hypatia_ridge_sc3, \
#     subj01_40sess_hypatia_ridge_flux, \
#     subj01_40sess_hypatia_ridge_sd35_no_blurry, \
#     subj01_40sess_hypatia_ridge_sc_flux_enhanced, \
#     subj01_40sess_hypatia_ridge_sc_sd35_enhanced, \
#     subj01_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30,
#     subj01_40sess_hypatia_ridge_sc_short_captions" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --output_file methods_scatter_reduced4

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     subj01_40sess_hypatia_ridge_sc3, \
#     subj01_40sess_hypatia_ridge_sc_vdvae, \
#     subj01_40sess_hypatia_ridge_sc_0.60_strength, \
#     subj01_40sess_hypatia_ridge_sc_0.65_strength, \
#     subj01_40sess_hypatia_ridge_sc_0.70_strength, \
#     subj01_40sess_hypatia_ridge_sc_0.80_strength" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --output_file methods_scatter_reduced6

# python plots_across_methods.py \
# --methods "subj05_40sess_hypatia_sc_nsd_general, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_15, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_16, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_17, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_18, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_19, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_20, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_21, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_22, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_23, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_25, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_26, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_27, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_28, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_29, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_31, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_32, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_33, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_34, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_35, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_36, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_37, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_38, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_39, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_40, \
# subj05_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_41" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s5_samplewise_sc \
# --gradient

# python plots_across_methods.py \
# --methods "subj04_40sess_hypatia_sc_nsd_general, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_1, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_2, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_3, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_4, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_5, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_6, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_7, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_8, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_9, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_10, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_11, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_12, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_13, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_14, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_15, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_16, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_17, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_18, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_19, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_20, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_21, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_22, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_23, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_24, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_25, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_26, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_27, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_28, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_29, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s4_samplewise_sc \
# --gradient

# python plots_across_methods.py \
# --methods "subj08_40sess_hypatia_sc_nsd_general, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_1, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_2, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_3, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_4, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_5, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_6, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_7, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_8, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_9, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_10, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_11, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_12, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_13, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_14, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_15, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_16, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_17, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_18, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_19, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_20, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_21, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_22, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_23, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_24, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_25, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_26, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_27, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_28, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_29, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_31, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_32, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_33, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_34, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_35, \
# subj08_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_36" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --output_file methods_comparison_roi_threshold_s8_samplewise_sc \
# --gradient

# python plots_across_methods.py \
# --methods "subj04_40sess_hypatia_sc_nsd_general, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_1, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_2, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_3, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_4, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_5, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_6, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_7, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_8, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_9, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_10, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_11, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_12, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_13, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_14, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_15, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_16, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_17, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_18, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_19, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_20, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_21, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_22, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_23, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_24, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_25, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_26, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_27, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_28, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_29, \
# subj04_40sess_hypatia_sc_ridge_rank_order_rois_samplewise_30" \
# --data_path ../dataset \
# --output_path ../figs/ \
# --subjs 3,4,6,8 \
# --output_file methods_comparison_roi_threshold_s3468_samplewise_sc \
# --gradient


# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     subj01_40sess_hypatia_ridge_sc_0.70_strength, \
#     subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae, \
#     subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_100000_wd_50000_max_it, \
#     subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_500000_wd_50000_max_it, \
#     subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_1000000_wd_50000_max_it, \
#     subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_100000_wd_200000_max_it, \
#     subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_500000_wd_200000_max_it, \
#     subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_1000000_wd_200000_max_it, \
#     subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_100000_wd_1000000_max_it, \
#     subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_500000_wd_1000000_max_it, \
#     subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_1000000_wd_1000000_max_it, \
#     subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_5000000_wd_50000_max_it, \
#     subj01_40sess_hypatia_ridge_sc_vdvae_0.70_strength_10000000_wd_50000_max_it" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --output_file methods_scatter_reduced4

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     subj01_40sess_hypatia_ridge_sc3, \
#     subj01_40sess_hypatia_ridge_scv_0.70_strength, \
#     subj01_40sess_hypatia_ridge_sc_0.70_strength_vdvae_contrast_sharpness, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_medium_captions" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "InceptionV3,CLIP,EffNet-B,SwAV,PixCorr,SSIM,AlexNet(2),AlexNet(5)" \
#     --output_file methods_scatter_best_no_bc


# python plots_across_methods.py \
#     --methods "subj03_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_50000_wd_fs_fcon, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_100000_wd_fs_fcon, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_250000_wd_fs_fcon, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_500000_wd_fs_fcon, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_750000_wd_fs_fcon, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_1000000_wd_fs_fcon" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_vdvae_weight_decay_s3

#     python plots_across_methods.py \
#     --methods "subj06_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_50000_wd_fs_fcon, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_100000_wd_fs_fcon, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_250000_wd_fs_fcon, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_500000_wd_fs_fcon, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_750000_wd_fs_fcon, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_1000000_wd_fs_fcon" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_vdvae_weight_decay_s6

# python plots_across_methods.py \
#     --methods "subj03_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_50000_wd_fs_fcon, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_100000_wd_fs_fcon, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_250000_wd_fs_fcon, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_500000_wd_fs_fcon, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_750000_wd_fs_fcon, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_1000000_wd_fs_fcon" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5)" \
#     --output_file methods_scatter_vdvae_weight_decay_s3_low_level

#     python plots_across_methods.py \
#     --methods "subj06_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_50000_wd_fs_fcon, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_100000_wd_fs_fcon, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_250000_wd_fs_fcon, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_500000_wd_fs_fcon, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_750000_wd_fs_fcon, \
#     subj06_40sess_hypatia_ridge_scv_0.70_strength_1000000_wd_fs_fcon" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5)" \
#     --output_file methods_scatter_vdvae_weight_decay_s6_low_level

#     # --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5)" \
#     # --criteria "InceptionV3,CLIP,EffNet-B,SwAV" \
#     --criteria "Brain Corr. nsd_general,Brain Corr. V1,Brain Corr. V2,Brain Corr. V3,Brain Corr. V4,Brain Corr. higher_vis" \


# python plots_across_methods.py \
#     --methods "subj01_40sess_hypatia_ridge_scv_RO_rois_10, \
#     subj01_40sess_hypatia_ridge_scv_RO_rois_11, \
#     subj01_40sess_hypatia_ridge_scv_RO_rois_12, \
#     subj01_40sess_hypatia_ridge_scv_RO_rois_13, \
#     subj01_40sess_hypatia_ridge_scv_RO_rois_14, \
#     subj01_40sess_hypatia_ridge_scv_RO_rois_15" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_vdvae_RO_s1

# python plots_across_methods.py \
#     --methods "subj01_40sess_hypatia_ridge_scv_RO_rois_10, \
#     subj01_40sess_hypatia_ridge_scv_RO_rois_11, \
#     subj01_40sess_hypatia_ridge_scv_RO_rois_12, \
#     subj01_40sess_hypatia_ridge_scv_RO_rois_13, \
#     subj01_40sess_hypatia_ridge_scv_RO_rois_14, \
#     subj01_40sess_hypatia_ridge_scv_RO_rois_15, \
#     subj01_40sess_hypatia_ridge_scv_RO_rois_16" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5)" \
#     --output_file methods_scatter_vdvae_RO_s1_low_level


# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_4, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_8, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_16, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_4_retrieval, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_8_retrieval, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_16_retrieval" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_retrieval_high_level

# python plots_acro

# python plots_across_methods.py \
#     --methods "subj03_40sess_hypatia_ridge_scv_0.70_strength_100000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_50000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_10000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_1000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_100_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_10_vwd_fs_fcon_short_captions" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --subjs 3,4 \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5)" \
#     --output_file methods_scatter_vdvae_weight_decay2_s34_low_level

#     python plots_across_methods.py \
#     --methods "subj03_40sess_hypatia_ridge_scv_0.70_strength_100000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_50000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_10000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_1000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_100_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_10_vwd_fs_fcon_short_captions" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --subjs 3,4 \
#     --criteria "InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_vdvae_weight_decay2_s34_high_level

#     python plots_across_methods.py \
#     --methods "subj03_40sess_hypatia_ridge_scv_0.70_strength_100000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_50000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_10000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_1000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_100_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_10_vwd_fs_fcon_short_captions" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --subjs 3,4 \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_vdvae_weight_decay2_s34



# python plots_across_methods.py \
#     --methods "subj03_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_snr_0.70, \
#     subj03_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_snr_0.65, \
#     subj03_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_snr_0.60, \
#     subj03_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_snr_0.55, \
#     subj03_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_snr_0.50, \
#     subj03_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_snr_0.45, \
#     subj03_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_snr_0.40, \
#     subj03_40sess_hypatia_ridge_scv_0.70_strength_100000_vwd_fs_fcon_short_captions" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_snr_s3

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_4_retrieval, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_8_retrieval, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_16_retrieval" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_reduced_retrieval

# python plots_across_methods.py \
#     --methods "subj03_40sess_hypatia_ridge_scv_vdvae_only_nsd_general, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_1, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_2, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_3, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_4, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_5, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_6, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_7, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_8, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_9, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_10, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_11, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_12, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_13, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_14, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_15, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_16, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_17, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_18, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_19, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_20, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_21, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_22, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_23, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_24, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_25, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_26, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_27, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_28, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_29, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_30, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_31, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_32, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_33, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_34, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_35, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_36, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_37, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_38, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_39, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_40, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_41, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_42, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_43, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_44, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_45, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_46, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_47, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_48, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_49, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_50, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_51, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_52, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_53, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_54, \
#     subj03_40sess_hypatia_ridge_scv_vdvae_only_55" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5)" \
#     --gradient \
#     --output_file methods_scatter_vdvae_rank_order_s3

# python plots_across_methods.py \
#     --methods "subj03_40sess_hypatia_ridge_scv_0.70_strength_100000_vwd_fs_fcon_short_captions, \
#     subj03_40sess_hypatia_ridge_scv_nsd_general_RO_rois_2, \
#     subj03_40sess_hypatia_ridge_scv_nsd_general_RO_rois_3, \
#     subj03_40sess_hypatia_ridge_scv_nsd_general_RO_rois_4, \
#     subj03_40sess_hypatia_ridge_scv_nsd_general_RO_rois_5, \
#     subj03_40sess_hypatia_ridge_scv_nsd_general_RO_rois_7" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_nsd_general_rank_order_s3

# python plots_across_methods.py \
#     --methods "subj06_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj06_40sess_hypatia_ridge_scv_nsd_general_RO_rois_2, \
#     subj06_40sess_hypatia_ridge_scv_nsd_general_RO_rois_3, \
#     subj06_40sess_hypatia_ridge_scv_nsd_general_RO_rois_4, \
#     subj06_40sess_hypatia_ridge_scv_nsd_general_RO_rois_5, \
#     subj06_40sess_hypatia_ridge_scv_nsd_general_RO_rois_6, \
#     subj06_40sess_hypatia_ridge_scv_nsd_general_RO_rois_7" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_nsd_general_rank_order_s6

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_16_retrieval, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_2, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_3, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_4, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_5, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_6, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_7" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_nsd_general_rank_order_s1

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_16_retrieval, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_2, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_3, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_4, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_5, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_6, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_7" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_nsd_general_rank_order_s1_high_level

# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions_ni_16_retrieval, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_2, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_3, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_4, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_5, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_6, \
#     subj01_40sess_hypatia_ridge_scv_nsd_general_RO_rois_7" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5)" \
#     --output_file methods_scatter_nsd_general_rank_order_s1_low_level


# python plots_across_methods.py \
#     --methods "mindeye1_subj01, \
#     braindiffuser_subj01, \
#     final_subj01_pretrained_40sess_24bs, \
#     subj01_40sess_hypatia_ridge_scv_0.70_strength_100000_wd_fs_fcon, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_short_captions, \
#     subj01_40sess_hypatia_ridge_svc_0.70_strength_fs_fcon_medium_captions, \
#     takagi_subj01, \
#     iCNN_subj01, \
#     subj01_40sess_hypatia_mirage_schmedium_train_medium_normalize, \
#     subj01_40sess_hypatia_mirage_no_retrieval, \
#     subj01_40sess_hypatia_mirage2, \
#     subj01_40sess_hypatia_mirage3, \
#     subj01_40sess_hypatia_mirage5, \
#     subj01_40sess_hypatia_mirage4_no_retrieval, \
#     subj01_40sess_hypatia_mirage" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_final_s1


# python plots_across_methods.py \
#     --methods "mindeye1_subj02, \
#     braindiffuser_subj02, \
#     final_subj02_pretrained_40sess_24bs, \
#     takagi_subj02, \
#     iCNN_subj02, \
#     subj02_40sess_hypatia_mirage2, \
#     subj05_40sess_hypatia_mirage5, \
#     subj02_40sess_hypatia_mirage_no_retrieval, \
#     subj02_40sess_hypatia_mirage" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_final_s2

# python plots_across_methods.py \
#     --methods "mindeye1_subj05, \
#     braindiffuser_subj05, \
#     final_subj05_pretrained_40sess_24bs, \
#     takagi_subj05, \
#     iCNN_subj05, \
#     subj05_40sess_hypatia_mirage2, \
#     subj05_40sess_hypatia_mirage5, \
#     subj05_40sess_hypatia_mirage_no_retrieval, \
#     subj05_40sess_hypatia_mirage" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_final_s5
    

# python plots_across_methods.py \
#     --methods "mindeye1_subj07, \
#     braindiffuser_subj07, \
#     final_subj07_pretrained_40sess_24bs, \
#     takagi_subj07, \
#     iCNN_subj07, \
#     subj07_40sess_hypatia_mirage2, \
#     subj07_40sess_hypatia_mirage_no_retrieval, \
#     subj07_40sess_hypatia_mirage" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_final_s7

# python plots_across_methods.py \
#     --methods "mindeye1_subj07, \
#     braindiffuser_subj07, \
#     final_subj07_pretrained_40sess_24bs, \
#     takagi_subj07, \
#     iCNN_subj07, \
#     subj07_40sess_hypatia_mirage2, \
#     subj07_40sess_hypatia_mirage_no_retrieval, \
#     subj07_40sess_hypatia_mirage" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --subjs 1,2,5,7 \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_final

# python plots_across_methods.py \
#     --methods "subj06_40sess_hypatia_mirage_weight_decay_1_abalation, \
#     subj06_40sess_hypatia_mirage_weight_decay_10_abalation, \
#     subj06_40sess_hypatia_mirage_weight_decay_100_abalation, \
#     subj06_40sess_hypatia_mirage_weight_decay_1000_abalation, \
#     subj06_40sess_hypatia_mirage_weight_decay_10000_abalation, \
#     subj06_40sess_hypatia_mirage_weight_decay_100000_abalation, \
#     subj06_40sess_hypatia_mirage_weight_decay_1000000_abalation, \
#     subj06_40sess_hypatia_mirage_weight_decay_10000000_abalation" \
#     --data_path ../dataset \
#     --output_path ../figs/ \
#     --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
#     --output_file methods_scatter_ridge_ablation_s6

python plots_across_methods.py \
    --methods "subj03_40sess_hypatia_mirage_weight_decay_1_abalation, \
    subj03_40sess_hypatia_mirage_weight_decay_10_abalation, \
    subj03_40sess_hypatia_mirage_weight_decay_100_abalation, \
    subj03_40sess_hypatia_mirage_weight_decay_1000_abalation, \
    subj03_40sess_hypatia_mirage_weight_decay_10000_abalation, \
    subj03_40sess_hypatia_mirage_weight_decay_100000_abalation, \
    subj03_40sess_hypatia_mirage_weight_decay_1000000_abalation, \
    subj03_40sess_hypatia_mirage_weight_decay_10000000_abalation" \
    --data_path ../dataset \
    --output_path ../figs/ \
    --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
    --output_file methods_scatter_ridge_ablation_s3

    python plots_across_methods.py \
    --methods "subj08_40sess_hypatia_mirage_weight_decay_1_abalation, \
    subj08_40sess_hypatia_mirage_weight_decay_1000_abalation, \
    subj08_40sess_hypatia_mirage_weight_decay_10000_abalation, \
    subj08_40sess_hypatia_mirage_weight_decay_100000_abalation, \
    subj08_40sess_hypatia_mirage_weight_decay_1000000_abalation, \
    subj08_40sess_hypatia_mirage_weight_decay_10000000_abalation" \
    --data_path ../dataset \
    --output_path ../figs/ \
    --criteria "PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV" \
    --output_file methods_scatter_ridge_ablation_s8