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


betas = utils.create_snr_betas(subject=subj, data_type=torch.float16, data_path=data_path, threshold = snr_threshold)
x_train, valid_nsd_ids_train, x_test, test_nsd_ids = utils.load_nsd(subject=subj, betas=betas, data_path=data_path)
betas_ro = utils.load_subject_based_on_rank_order_rois(excluded_subject=subj, data_type=torch.float16, top_n_rois=top_n_rank_order_rois, samplewise=samplewise_rank_order_rois)
x_train_ro, _, _, _ = utils.load_nsd(subject=subj, betas=betas_ro, data_path=data_path)
print(x_train.shape, valid_nsd_ids_train.shape)

print(f"Loaded subj {subj} betas!\n")


# In[ ]:


# Load 73k NSD images
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images'] # if you go OOM you can remove the [:] so it isnt preloaded to cpu! (will require a few edits elsewhere tho)
# images = torch.Tensor(images).to("cpu").to(data_type)
print("Loaded all 73k possible NSD images to cpu!", images.shape)

# Load 73k NSD captions
if caption_type == "coco":
    caption_file = "annots_73k.npy"
elif caption_type == "short":
    caption_file = "short_length_captions.npy"
elif caption_type == "medium":
    caption_file = "mid_length_caps.npy"
else:
    raise ValueError("Invalid caption type")
captions = np.load(f'{data_path}/preprocessed_data/{caption_file}')
print("Loaded all 73k NSD captions to cpu!", captions.shape)

train_images = torch.zeros((len(valid_nsd_ids_train), 3, 224, 224))
train_captions = np.zeros((len(valid_nsd_ids_train),), dtype=object)
for i, idx in enumerate(valid_nsd_ids_train):
    train_images[i] =  torch.from_numpy(images[idx])
    train_captions[i] = captions[idx]
print(f"Filtered down to only the {len(valid_nsd_ids_train)} training images for subject {subj}!")


# ## Load models

# ### Feature extractor model

# In[ ]:


clip_extractor = SC_Reconstructor(compile_models=True, embedder_only=True, device=device, cache_dir=cache_dir)
vdvae = VDVAE(device=device, cache_dir=cache_dir)
image_embedding_variant = "stable_cascade"
clip_emb_dim = 768
clip_seq_dim = 1
text_embedding_variant = "stable_cascade"
clip_text_seq_dim=77
clip_text_emb_dim=1280
latent_embedding_variant = "vdvae"
latent_emb_dim = 91168

if caption_type != "coco":
    text_embedding_variant += f"_{caption_type}"


# # Creating block of CLIP embeddings

# In[ ]:


file_path = f"{data_path}/preprocessed_data/subject{subj}/{image_embedding_variant}_image_embeddings_train.pt"
emb_batch_size = 50
if not os.path.exists(file_path):
    # Generate CLIP Image embeddings
    print("Generating Image embeddings!")
    clip_image_train = torch.zeros((len(train_images), clip_seq_dim, clip_emb_dim)).to("cpu")
    for i in tqdm(range(len(train_images) // emb_batch_size), desc="Encoding images..."):
        batch_list = []
        for img in train_images[i * emb_batch_size:i * emb_batch_size + emb_batch_size]:
            batch_list.append(transforms.ToPILImage()(img))
        clip_image_train[i * emb_batch_size:i * emb_batch_size + emb_batch_size] = clip_extractor.embed_image(batch_list).to("cpu")

    torch.save(clip_image_train, file_path)
else:
    clip_image_train = torch.load(file_path)

        
if dual_guidance:
    emb_batch_size = 50
    file_path_txt = f"{data_path}/preprocessed_data/subject{subj}/{text_embedding_variant}_text_embeddings_train.pt"
    if not os.path.exists(file_path_txt):
        # Generate CLIP Text embeddings
        print("Generating Text embeddings!")
        clip_text_train = torch.zeros((len(train_captions), clip_text_seq_dim, clip_text_emb_dim)).to("cpu")
        for i in tqdm(range(len(train_captions) // emb_batch_size), desc="Encoding captions..."):
            batch_captions = train_captions[i * emb_batch_size:i * emb_batch_size + emb_batch_size].tolist()
            clip_text_train[i * emb_batch_size:i * emb_batch_size + emb_batch_size] =  clip_extractor.embed_text(batch_captions).to("cpu")
        torch.save(clip_text_train, file_path_txt)
    else:
        clip_text_train = torch.load(file_path_txt)


if blurry_recon:
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
print(f"Loaded train image clip {clip_image_train.shape}, text clip {clip_text_train.shape}, and VAE {vae_image_train.shape} for subj{subj}!", )


# # Train Ridge regression models

# In[ ]:


start = time.time()
ridge_weights = np.zeros((clip_seq_dim * clip_emb_dim, x_train.shape[-1])).astype(np.float32)
ridge_biases = np.zeros((clip_seq_dim * clip_emb_dim)).astype(np.float32)
print(f"Training Ridge Image model with alpha=100000")
model = Ridge(
    alpha=100000,
    max_iter=max_iter,
    random_state=42,
)

model.fit(x_train, clip_image_train.reshape(len(clip_image_train), -1))
ridge_weights = model.coef_
ridge_biases = model.intercept_
datadict = {"coef" : ridge_weights, "intercept" : ridge_biases}
# Save the regression weights
with open(f'{outdir}/ridge_image_weights.pkl', 'wb') as f:
    pickle.dump(datadict, f)
    
if dual_guidance:
    ridge_weights_txt = np.zeros((clip_text_seq_dim * clip_text_emb_dim, x_train.shape[-1])).astype(np.float32)
    ridge_biases_txt = np.zeros((clip_text_seq_dim * clip_text_emb_dim)).astype(np.float32)
    print(f"Training Ridge Text model with alpha=100000")
    model = Ridge(
        alpha=100000,
        max_iter=max_iter,
        random_state=42,
    )

    model.fit(x_train, clip_text_train.reshape(len(clip_text_train), -1))
    ridge_weights_txt = model.coef_
    ridge_biases_txt = model.intercept_
    datadict = {"coef" : ridge_weights_txt, "intercept" : ridge_biases_txt}
    # Save the regression weights
    with open(f'{outdir}/ridge_text_weights.pkl', 'wb') as f:
        pickle.dump(datadict, f)
            
if blurry_recon:
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

