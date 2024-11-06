#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import json
import pickle
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
import PIL
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image, ImageDraw, ImageFont, ImageEnhance

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
# sys.path.append('generative_models/')
# import sgm
from sc_reconstructor import SC_Reconstructor
from vdvae import VDVAE
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
# from models import *
device = "cuda"
print("device:",device)


# In[ ]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    # model_name = "final_subj01_pretrained_40sess_24bs"
    model_name = "subj01_40sess_hypatia_ridge_sc2"
    print("model_name:", model_name)

    # other variables can be specified in the following string:
    jupyter_args = f"--data_path=../dataset \
                    --cache_dir=../cache \
                    --model_name={model_name} --subj=1 \
                    --mode vision \
                    --dual_guidance"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in ../train_logs/model_name",
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
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=2048,
)
parser.add_argument(
    "--seq_len",type=int,default=1,
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--mode",type=str,default="vision",
)
parser.add_argument(
    "--gen_rep",type=int,default=10,
)
parser.add_argument(
    "--dual_guidance",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--snr",type=float,default=-1,
)
parser.add_argument(
    "--normalize_preds",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--save_raw",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--raw_path",type=str,
)
parser.add_argument(
    "--strength",type=float,default=0.70,
)
parser.add_argument(
    "--textstrength",type=float,default=0.5,
)
parser.add_argument(
    "--top_n_rank_order_rois",type=int, default=-1,
    help="Used for selecting the top n rois on a whole brain to narrow down voxels.",
)
parser.add_argument(
    "--samplewise_rank_order_rois",action=argparse.BooleanOptionalAction,default=False,
    help="Use the samplewise rank order rois versus voxelwise",
)
parser.add_argument(
    "--vdvae",action=argparse.BooleanOptionalAction, default=False,
    help="Use the braindiffuser VDVAE as the low level image",
)
parser.add_argument(
    "--filter_contrast",action=argparse.BooleanOptionalAction, default=True,
    help="Filter the low level output to be more intense and smoothed",
)
parser.add_argument(
    "--filter_sharpness",action=argparse.BooleanOptionalAction, default=True,
    help="Filter the low level output to be more intense and smoothed",
)
parser.add_argument(
    "--filter_color",action=argparse.BooleanOptionalAction, default=False,
    help="Filter the low level output to be more intense and smoothed",
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)


if seed > 0 and gen_rep == 1:
    # seed all random functions, but only if doing 1 rep
    utils.seed_everything(seed)
    

outdir = os.path.abspath(f'../train_logs/{model_name}')

# make output directory
os.makedirs("evals",exist_ok=True)
os.makedirs(f"evals/{model_name}",exist_ok=True)


# # Load data

# In[ ]:


if mode == "synthetic":
    voxels, all_images = utils.load_nsd_synthetic(subject=subj, average=False, nest=True)
elif subj > 8:
    _, _, voxels, all_images = utils.load_imageryrf(subject=subj-8, mode=mode, stimtype="object", average=False, nest=True, split=True)
else:
    voxels, all_images = utils.load_nsd_mental_imagery(subject=subj, mode=mode, stimtype="all", average=True, nest=False)
    voxels_ro, all_images = utils.load_nsd_mental_imagery(subject=subj, mode=mode, stimtype="all", average=True, nest=False, top_n_rois=top_n_rank_order_rois, samplewise=samplewise_rank_order_rois)
    #, top_n_rois=top_n_rank_order_rois, samplewise=samplewise_rank_order_rois
num_voxels = voxels.shape[-1]


# # Load pretrained models

# # Load Stable Cascade

# In[ ]:


reconstructor = SC_Reconstructor(compile_models=False, device=device)
vdvae = VDVAE(device=device, cache_dir=cache_dir)
image_embedding_variant = "stable_cascade"
clip_emb_dim = 768
clip_seq_dim = 1
text_embedding_variant = "stable_cascade"
clip_text_seq_dim=77
clip_text_emb_dim=1280
latent_embedding_variant = "vdvae"
latent_emb_dim = 91168


# ### Compute ground truth embeddings for training data (for feature normalization)

# In[ ]:


# If this is erroring, feature extraction failed in Train.ipynb
if normalize_preds:
    file_path = f"{data_path}/preprocessed_data/subject{subj}/{image_embedding_variant}_image_embeddings_train.pt"
    clip_image_train = torch.load(file_path)
        
    if dual_guidance:
        file_path_txt = f"{data_path}/preprocessed_data/subject{subj}/{text_embedding_variant}_text_embeddings_train.pt"
        clip_text_train = torch.load(file_path_txt)
        
    if blurry_recon:
        file_path = f"{data_path}/preprocessed_data/subject{subj}/{latent_embedding_variant}_latent_embeddings_train.pt"
        vae_image_train = torch.load(file_path)
    else:
        strength = 1.0


# # Predicting latent vectors for reconstruction  

# In[ ]:


pred_clip_image = torch.zeros((len(all_images), clip_seq_dim, clip_emb_dim)).to("cpu")
with open(f'{outdir}/ridge_image_weights.pkl', 'rb') as f:
    image_datadict = pickle.load(f)
model = Ridge(
    alpha=60000,
    max_iter=50000,
    random_state=42,
)
model.coef_ = image_datadict["coef"]
model.intercept_ = image_datadict["intercept"]
pred_clip_image = torch.from_numpy(model.predict(voxels[:,0]).reshape(-1, clip_seq_dim, clip_emb_dim))

if dual_guidance:
    with open(f'{outdir}/ridge_text_weights.pkl', 'rb') as f:
        text_datadict = pickle.load(f)
    pred_clip_text = torch.zeros((len(all_images), clip_text_seq_dim, clip_text_emb_dim)).to("cpu")
    model = Ridge(
        alpha=60000,
        max_iter=50000,
        random_state=42,
    )
    model.coef_ = text_datadict["coef"]
    model.intercept_ = text_datadict["intercept"]
    pred_clip_text = torch.from_numpy(model.predict(voxels[:,0]).reshape(-1, clip_text_seq_dim, clip_text_emb_dim))
    
if blurry_recon:
    pred_blurry_vae = torch.zeros((len(all_images), latent_emb_dim)).to("cpu")
    with open(f'{outdir}/ridge_blurry_weights.pkl', 'rb') as f:
        blurry_datadict = pickle.load(f)
    model = Ridge(
        alpha=60000,
        max_iter=50000,
        random_state=42,
    )
    model.coef_ = blurry_datadict["coef"]
    model.intercept_ = blurry_datadict["intercept"]
    pred_blurry_vae = torch.from_numpy(model.predict(voxels_ro[:,0]).reshape(-1, latent_emb_dim))    
    
if normalize_preds:
    std_pred_clip_image = (pred_clip_image - torch.mean(pred_clip_image,axis=0)) / (torch.std(pred_clip_image,axis=0) + 1e-6)
    pred_clip_image = std_pred_clip_image * torch.std(clip_image_train,axis=0) + torch.mean(clip_image_train,axis=0)
    if dual_guidance:
        std_pred_clip_text = (pred_clip_text - torch.mean(pred_clip_text,axis=0)) / (torch.std(pred_clip_text,axis=0) + 1e-6)
        pred_clip_text = std_pred_clip_text * torch.std(clip_text_train,axis=0) + torch.mean(clip_text_train,axis=0)
    if blurry_recon:
        std_pred_blurry_vae = (pred_blurry_vae - torch.mean(pred_blurry_vae,axis=0)) / (torch.std(pred_blurry_vae,axis=0) + 1e-6)
        pred_blurry_vae = std_pred_blurry_vae * torch.std(vae_image_train,axis=0) + torch.mean(vae_image_train,axis=0)


# In[ ]:


final_recons = None
final_predcaptions = None
final_clipvoxels = None
final_blurryrecons = None

if save_raw:
    raw_root = f"{raw_path}/{mode}/{model_name}/subject{subj}/"
    print("raw_root:", raw_root)
    os.makedirs(raw_root,exist_ok=True)
    torch.save(pred_clip_image, f"{raw_root}/{image_embedding_variant}_image_voxels.pt")
    if dual_guidance:
        torch.save(pred_clip_text, f"{raw_root}/{text_embedding_variant}_text_voxels.pt")
    if blurry_recon:
        torch.save(pred_blurry_vae, f"{raw_root}/{latent_embedding_variant}_latent_voxels.pt")


for idx in tqdm(range(0,voxels.shape[0]), desc="sample loop"):
    clip_voxels = pred_clip_image[idx]
    if dual_guidance:
        clip_text_voxels = pred_clip_text[idx]
    else:
        clip_text_voxels = None
    # Save retrieval submodule outputs
    if final_clipvoxels is None:
        final_clipvoxels = clip_voxels.unsqueeze(0).to('cpu')
    else:
        final_clipvoxels = torch.vstack((final_clipvoxels, clip_voxels.unsqueeze(0).to('cpu')))
    latent_voxels=None
    if blurry_recon:
        latent_voxels = pred_blurry_vae[idx].unsqueeze(0)
        blurred_image = vdvae.reconstruct(latents=latent_voxels)
        if filter_sharpness:
            # This helps make the output not blurry when using the VDVAE
            blurred_image = ImageEnhance.Sharpness(blurred_image).enhance(20)
        if filter_contrast:
            # This boosts the structural impact of the blurred_image
            blurred_image = ImageEnhance.Contrast(blurred_image).enhance(1.5)
        if filter_color: 
            blurred_image = ImageEnhance.Color(blurred_image).enhance(0.5)
        im = transforms.ToTensor()(blurred_image)
        if final_blurryrecons is None:
            final_blurryrecons = im.cpu()
        else:
            final_blurryrecons = torch.vstack((final_blurryrecons, im.cpu()))
                
    samples = reconstructor.reconstruct(image=blurred_image,
                                        c_i=clip_voxels,
                                        c_t=clip_text_voxels,
                                        n_samples=gen_rep,
                                        textstrength=textstrength,
                                        strength=strength)
    
    if save_raw:
        os.makedirs(f"{raw_root}/{idx}/", exist_ok=True)
        for rep in range(gen_rep):
            transforms.ToPILImage()(samples[rep]).save(f"{raw_root}/{idx}/{rep}.png")
        transforms.ToPILImage()(all_images[idx]).save(f"{raw_root}/{idx}/ground_truth.png")
        transforms.ToPILImage()(transforms.ToTensor()(blurred_image).cpu()).save(f"{raw_root}/{idx}/low_level.png")
        torch.save(clip_voxels, f"{raw_root}/{idx}/clip_image_voxels.pt")
        if dual_guidance:
            torch.save(clip_text_voxels, f"{raw_root}/{idx}/clip_text_voxels.pt")

    if final_recons is None:
        final_recons = samples.unsqueeze(0).cpu()
    else:
        final_recons = torch.cat((final_recons, samples.unsqueeze(0).cpu()), dim=0)
        
if blurry_recon:
    torch.save(final_blurryrecons,f"evals/{model_name}/{model_name}_all_blurryrecons_{mode}.pt")
torch.save(final_recons,f"evals/{model_name}/{model_name}_all_recons_{mode}.pt")
torch.save(final_clipvoxels,f"evals/{model_name}/{model_name}_all_clipvoxels_{mode}.pt")
print(f"saved {model_name} mi outputs!")


# In[ ]:


if not utils.is_interactive():
    sys.exit(0)

