#!/usr/bin/env python
# coding: utf-8

# # Import packages & functions

# In[2]:


import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds
import gc
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
from sklearn.linear_model import Ridge
import pickle
# custom functions #
import utils
from sc_reconstructor import SC_Reconstructor
from vdvae import VDVAE


# # Configurations

# In[ ]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "subj01_40sess_hypatia_ridge_scsubj01_40sess_hypatia_ridge_sc_medium_captions"
    print("model_name:", model_name)
    
    # global_batch_size and batch_size should already be defined in the 2nd cell block
    jupyter_args = f"--data_path=../dataset/ \
                    --cache_dir=../cache/ \
                    --model_name={model_name} \
                    --batch_size=64 \
                    --no-multi_subject --subj=1 --num_sessions=40 \
                    --dual_guidance --caption_type medium"

    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[4]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--data_path", type=str, default=os.getcwd(),
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default=os.getcwd(),
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8,9,10,11],
    help="Validate on which subject?",
)
parser.add_argument(
    "--multisubject_ckpt", type=str, default=None,
    help="Path to pre-trained multisubject model to finetune a single subject from. multisubject must be False.",
)
parser.add_argument(
    "--num_sessions", type=int, default=1,
    help="Number of training sessions to include",
)
parser.add_argument(
    "--use_prior",action=argparse.BooleanOptionalAction,default=True,
    help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
)
parser.add_argument(
    "--visualize_prior",action=argparse.BooleanOptionalAction,default=False,
    help="output visualizations from unCLIP every ckpt_interval (requires much more memory!)",
)
parser.add_argument(
    "--batch_size", type=int, default=16,
    help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
)
parser.add_argument(
    "--wandb_log",action=argparse.BooleanOptionalAction,default=False,
    help="whether to log to wandb",
)
parser.add_argument(
    "--resume_from_ckpt",action=argparse.BooleanOptionalAction,default=False,
    help="if not using wandb and want to resume from a ckpt",
)
parser.add_argument(
    "--wandb_project",type=str,default="stability",
    help="wandb project name",
)
parser.add_argument(
    "--mixup_pct",type=float,default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
    help="whether to output blurry reconstructions",
)
parser.add_argument(
    "--blur_scale",type=float,default=.5,
    help="multiply loss from blurry recons by this number",
)
parser.add_argument(
    "--clip_scale",type=float,default=1.,
    help="multiply contrastive loss by this number",
)
parser.add_argument(
    "--prior_scale",type=float,default=30,
    help="multiply diffusion prior loss by this",
)
parser.add_argument(
    "--use_image_aug",action=argparse.BooleanOptionalAction,default=False,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--num_epochs",type=int,default=150,
    help="number of epochs of training",
)
parser.add_argument(
    "--multi_subject",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--new_test",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=1024,
)
parser.add_argument(
    "--seq_past",type=int,default=0,
)
parser.add_argument(
    "--seq_future",type=int,default=0,
)
parser.add_argument(
    "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
)
parser.add_argument(
    "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--ckpt_interval",type=int,default=5,
    help="save backup ckpt and reconstruct every x epochs",
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--max_lr",type=float,default=3e-4,
)
parser.add_argument(
    "--weight_decay",type=int,default=100000,
)
parser.add_argument(
    "--max_iter",type=int,default=50000,
)
parser.add_argument(
    "--train_imageryrf",action=argparse.BooleanOptionalAction,default=False,
    help="Use the ImageryRF dataset for pretraining",
)
parser.add_argument(
    "--no_nsd",action=argparse.BooleanOptionalAction,default=False,
    help="Don't use the Natural Scenes Dataset for pretraining",
)
parser.add_argument(
    "--snr_threshold",type=float,default=-1.0,
    help="Used for calculating SNR on a whole brain to narrow down voxels.",
)
parser.add_argument(
    "--mode",type=str,default="all",
)
parser.add_argument(
    "--dual_guidance",action=argparse.BooleanOptionalAction,default=False,
    help="Use the decoded captions for dual guidance",
)
parser.add_argument(
    "--top_n_rank_order_rois",type=int, default=-1,
    help="Used for selecting the top n rois on a whole brain to narrow down voxels.",
)
parser.add_argument(
    "--samplewise_rank_order_rois",action=argparse.BooleanOptionalAction, default=False,
    help="Use the samplewise rank order rois versus voxelwise",
)
parser.add_argument(
    "--caption_type",type=str,default='coco',choices=['coco','short', 'medium'],
)
parser.add_argument(
    "--reduced_samplewise_rank_order_rois",action=argparse.BooleanOptionalAction, default=False,
    help="Use the reduced samplewise rank order rois",
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
# seed all random functions
utils.seed_everything(seed)

outdir = os.path.abspath(f'../train_logs/{model_name}')
os.makedirs(outdir,exist_ok=True)
device = "cuda"


# # Prep data, models, and dataloaders

# In[ ]:


betas_ro = utils.load_subject_based_on_rank_order_rois(excluded_subject=subj, data_type=torch.float16, top_n_rois=top_n_rank_order_rois, reduced=reduced_samplewise_rank_order_rois)
x_train_ro, valid_nsd_ids_train, _, _ = utils.load_nsd(subject=subj, betas=betas_ro, data_path=data_path)
# print(x_train.shape, valid_nsd_ids_train.shape)

print(f"Loaded subj {subj} betas!\n")


# In[ ]:


# Load 73k NSD images
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images'] # if you go OOM you can remove the [:] so it isnt preloaded to cpu! (will require a few edits elsewhere tho)
# images = torch.Tensor(images).to("cpu").to(data_type)
print("Loaded all 73k possible NSD images to cpu!", images.shape)

train_images = torch.zeros((len(valid_nsd_ids_train), 3, 224, 224))
for i, idx in enumerate(valid_nsd_ids_train):
    train_images[i] =  torch.from_numpy(images[idx])
# print(f"Filtered down to only the {len(valid_nsd_ids_train)} training images for subject {subj}!")


# ## Load models

# ### Feature extractor model

# In[ ]:


vdvae = VDVAE(device=device, cache_dir=cache_dir)
latent_embedding_variant = "vdvae"
latent_emb_dim = 91168


# # Creating block of CLIP embeddings

# In[ ]:


emb_batch_size = 1
file_path = f"{data_path}/preprocessed_data/subject{subj}/{latent_embedding_variant}_latent_embeddings_train.pt"
if not os.path.exists(file_path):
    print("Generating Latent Image embeddings!")
    vae_image_train = torch.zeros((len(train_images), latent_emb_dim)).to("cpu")
    for i in tqdm(range(len(train_images)), desc="Encoding images..."):
        img = transforms.ToPILImage()(train_images[i])
        vae_image_train[i * emb_batch_size:i * emb_batch_size + emb_batch_size] = vdvae.embed_latent(img).reshape(-1, latent_emb_dim).to("cpu")
    torch.save(vae_image_train, file_path)
else:
    vae_image_train = torch.load(file_path)
print(f"Loaded train VAE {vae_image_train.shape} for subj{subj}!")


# # Train Ridge regression models

# In[ ]:


start = time.time()
            

ridge_weights_blurry = np.zeros((latent_emb_dim, x_train_ro.shape[-1])).astype(np.float32)
ridge_biases_blurry = np.zeros((latent_emb_dim,)).astype(np.float32)
print(f"Training Ridge Blurry recon model with alpha={weight_decay}")
model = Ridge(
    alpha=weight_decay,
    max_iter=max_iter,
    random_state=42,
)
model.fit(x_train_ro, vae_image_train)
ridge_weights_blurry = model.coef_
ridge_biases_blurry = model.intercept_
datadict = {"coef" : ridge_weights_blurry, "intercept" : ridge_biases_blurry}
# Save the regression weights
with open(f'{outdir}/ridge_blurry_weights.pkl', 'wb') as f:
    pickle.dump(datadict, f)


print(f"Elapsed training time for {model_name}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


# In[ ]:


#voxels_ro, all_images = utils.load_nsd_mental_imagery(subject=subj, mode=mode, stimtype="all", average=True, nest=False, top_n_rois=top_n_rank_order_rois, samplewise=samplewise_rank_order_rois)


# In[ ]:


# reconstructor = SC_Reconstructor(compile_models=False, device=device)
# vdvae = VDVAE(device=device, cache_dir=cache_dir)
# latent_embedding_variant = "vdvae"
# latent_emb_dim = 91168


# In[ ]:


# # If this is erroring, feature extraction failed in Train.ipynb
# if normalize_preds:
        
#     if blurry_recon:
#         file_path = f"{data_path}/preprocessed_data/subject{subj}/{latent_embedding_variant}_latent_embeddings_train.pt"
#         vae_image_train = torch.load(file_path)
#     else:
#         strength = 1.0


# In[ ]:


# if blurry_recon:
#     pred_blurry_vae = torch.zeros((len(all_images), latent_emb_dim)).to("cpu")
#     with open(f'{outdir}/ridge_blurry_weights.pkl', 'rb') as f:
#         blurry_datadict = pickle.load(f)
#     model = Ridge(
#         alpha=60000,
#         max_iter=50000,
#         random_state=42,
#     )
#     model.coef_ = blurry_datadict["coef"]
#     model.intercept_ = blurry_datadict["intercept"]
#     pred_blurry_vae = torch.from_numpy(model.predict(voxels_ro[:,0]).reshape(-1, latent_emb_dim))    
    
# if normalize_preds:
#     if blurry_recon:
#         std_pred_blurry_vae = (pred_blurry_vae - torch.mean(pred_blurry_vae,axis=0)) / (torch.std(pred_blurry_vae,axis=0) + 1e-6)
#         pred_blurry_vae = std_pred_blurry_vae * torch.std(vae_image_train,axis=0) + torch.mean(vae_image_train,axis=0)


# In[ ]:


# final_recons = None
# final_predcaptions = None
# final_clipvoxels = None
# final_blurryrecons = None

# if save_raw:
#     raw_root = f"{raw_path}/{mode}/{model_name}/subject{subj}/"
#     print("raw_root:", raw_root)
#     os.makedirs(raw_root,exist_ok=True)
#     torch.save(pred_clip_image, f"{raw_root}/{image_embedding_variant}_image_voxels.pt")
#     if dual_guidance:
#         torch.save(pred_clip_text, f"{raw_root}/{text_embedding_variant}_text_voxels.pt")
#     if blurry_recon:
#         torch.save(pred_blurry_vae, f"{raw_root}/{latent_embedding_variant}_latent_voxels.pt")


# for idx in tqdm(range(0,voxels.shape[0]), desc="sample loop"):
#     clip_voxels = pred_clip_image[idx]
#     if dual_guidance:
#         clip_text_voxels = pred_clip_text[idx]
#     else:
#         clip_text_voxels = None
#     # Save retrieval submodule outputs
#     if final_clipvoxels is None:
#         final_clipvoxels = clip_voxels.unsqueeze(0).to('cpu')
#     else:
#         final_clipvoxels = torch.vstack((final_clipvoxels, clip_voxels.unsqueeze(0).to('cpu')))
#     latent_voxels=None
#     if blurry_recon:
#         latent_voxels = pred_blurry_vae[idx].unsqueeze(0)
#         blurred_image = vdvae.reconstruct(latents=latent_voxels)
#         if filter_sharpness:
#             # This helps make the output not blurry when using the VDVAE
#             blurred_image = ImageEnhance.Sharpness(blurred_image).enhance(20)
#         if filter_contrast:
#             # This boosts the structural impact of the blurred_image
#             blurred_image = ImageEnhance.Contrast(blurred_image).enhance(1.5)
#         if filter_color: 
#             blurred_image = ImageEnhance.Color(blurred_image).enhance(0.5)
#         im = transforms.ToTensor()(blurred_image)
#         if final_blurryrecons is None:
#             final_blurryrecons = im.cpu()
#         else:
#             final_blurryrecons = torch.vstack((final_blurryrecons, im.cpu()))
                
#     samples = reconstructor.reconstruct(image=blurred_image,
#                                         c_i=clip_voxels,
#                                         c_t=clip_text_voxels,
#                                         n_samples=gen_rep,
#                                         textstrength=textstrength,
#                                         strength=strength)
    
#     if save_raw:
#         os.makedirs(f"{raw_root}/{idx}/", exist_ok=True)
#         for rep in range(gen_rep):
#             transforms.ToPILImage()(samples[rep]).save(f"{raw_root}/{idx}/{rep}.png")
#         transforms.ToPILImage()(all_images[idx]).save(f"{raw_root}/{idx}/ground_truth.png")
#         transforms.ToPILImage()(transforms.ToTensor()(blurred_image).cpu()).save(f"{raw_root}/{idx}/low_level.png")
#         torch.save(clip_voxels, f"{raw_root}/{idx}/clip_image_voxels.pt")
#         if dual_guidance:
#             torch.save(clip_text_voxels, f"{raw_root}/{idx}/clip_text_voxels.pt")

#     if final_recons is None:
#         final_recons = samples.unsqueeze(0).cpu()
#     else:
#         final_recons = torch.cat((final_recons, samples.unsqueeze(0).cpu()), dim=0)
        
# if blurry_recon:
#     torch.save(final_blurryrecons,f"evals/{model_name}/{model_name}_all_blurryrecons_{mode}.pt")
# torch.save(final_recons,f"evals/{model_name}/{model_name}_all_recons_{mode}.pt")
# torch.save(final_clipvoxels,f"evals/{model_name}/{model_name}_all_clipvoxels_{mode}.pt")
# print(f"saved {model_name} mi outputs!")

