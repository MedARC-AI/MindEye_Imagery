#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

from PIL import Image, ImageDraw, ImageFont

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge
from versatile_diffusion import Reconstructor
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
from models import *

### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:",device)


# In[2]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    # model_name = "final_subj01_pretrained_40sess_24bs"
    model_name = "pretrained_subj01_40sess_hypatia_no_blurry2"
    print("model_name:", model_name)

    # other variables can be specified in the following string:
    jupyter_args = f"--data_path=../dataset \
                    --cache_dir=../cache \
                    --model_name={model_name} --subj=1 \
                    --hidden_dim=1024 --n_blocks=4 --mode imagery --no-blurry_recon"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[3]:


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
    "--deprecated",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--top_n_rank_order_rois",type=int, default=-1,
    help="Used for selecting the top n rois on a whole brain to narrow down voxels.",
)
parser.add_argument(
    "--samplewise_rank_order_rois",action=argparse.BooleanOptionalAction,default=False,
    help="Use the samplewise rank order rois versus voxelwise",
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

# In[4]:


if mode == "synthetic":
    voxels, all_images = utils.load_nsd_synthetic(subject=subj, average=False, nest=True)
elif subj > 8:
    _, _, voxels, all_images = utils.load_imageryrf(subject=subj-8, mode=mode, stimtype="object", average=False, nest=True, split=True)
else:
    voxels, all_images = utils.load_nsd_mental_imagery(subject=subj, mode=mode, stimtype="all", top_n_rois=top_n_rank_order_rois, samplewise=samplewise_rank_order_rois, average=True, nest=False)
num_voxels = voxels.shape[-1]


# # Load pretrained models

# In[ ]:


if blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')
    
    autoenc.load_state_dict(ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)
    


# # Load Versatile Diffusion

# In[ ]:


clip_emb_dim = 768
clip_seq_dim = 257
clip_text_seq_dim=77
reconstructor = Reconstructor(device=device, cache_dir=cache_dir, deprecated=deprecated)
clip_extractor = reconstructor
clip_variant = "ViT-L-14"


# ### Compute ground truth embeddings for training data (for feature normalization)

# In[ ]:


if normalize_preds:
    file_path = f"{data_path}/preprocessed_data/subject{subj}/{clip_variant}_image_embeddings_train.pt"
    clip_image_train = torch.load(file_path)
        
    if dual_guidance:
        file_path_txt = f"{data_path}/preprocessed_data/subject{subj}/{clip_variant}_text_embeddings_train.pt"
        clip_text_train = torch.load(file_path_txt)
            
    if blurry_recon:
        file_path = f"{data_path}/preprocessed_data/subject{subj}/autoenc_image_embeddings_train.pt"
        vae_image_train = torch.load(file_path)


# ### Compute ground truth embeddings for test data

# In[ ]:


file_path = f"{data_path}/preprocessed_data/subject{subj}/{clip_variant}_image_embeddings_{mode}.pt"

images_test = transforms.Resize((224, 224))(all_images)
text_test = np.load(f"{data_path}/preprocessed_data/captions_18.npy")
if not os.path.exists(file_path):
    # Generate CLIP Image embeddings
    print("Generating CLIP Image embeddings!")
    clip_image_test = torch.zeros((len(images_test), clip_seq_dim, clip_emb_dim)).to("cpu")
    for i in tqdm(range(0, len(images_test)), desc="Encoding images..."):
        clip_image_test[i] = clip_extractor.embed_image(images_test[i].unsqueeze(0)).detach().to("cpu")
    torch.save(clip_image_test, file_path)
else:
    clip_image_test = torch.load(file_path)
    
if dual_guidance:
    file_path_txt = f"{data_path}/preprocessed_data/subject{subj}/{clip_variant}_text_embeddings_{mode}.pt"
    if not os.path.exists(file_path_txt):
        # Generate CLIP Image embeddings
        print("Generating CLIP Text embeddings!")
        clip_text_test = torch.zeros((len(text_test), clip_text_seq_dim, clip_emb_dim)).to("cpu")
        for i in tqdm(range(0, len(text_test)), desc="Encoding captions..."):
            clip_text_test[i] = clip_extractor.embed_text(text_test[i]).detach().to("cpu")
        torch.save(clip_text_test, file_path_txt)
    else:
        clip_text_test = torch.load(file_path_txt)
if blurry_recon:
    file_path = f"{data_path}/preprocessed_data/subject{subj}/autoenc_image_embeddings_{mode}.pt"

    if not os.path.exists(file_path):
        # Generate CLIP Image embeddings
        print("Generating VAE Image embeddings!")
        vae_image_test = torch.zeros((len(images_test), 3136)).to("cpu")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for i in tqdm(range(0, len(images_test)), desc="Encoding images..."):
                vae_image_test[i] = (autoenc.encode(2*images_test[i].unsqueeze(0).detach().to(device=device, dtype=torch.float16)-1).latent_dist.mode() * 0.18215).detach().to("cpu").flatten()
            torch.save(vae_image_test, file_path)
    else:
        vae_image_test = torch.load(file_path)


# # Predicting latent vectors for reconstruction  

# In[ ]:


pred_clip_image = torch.zeros((len(images_test), clip_seq_dim, clip_emb_dim)).to("cpu")
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
    pred_clip_text = torch.zeros((len(text_test), clip_text_seq_dim, clip_emb_dim)).to("cpu")
    model = Ridge(
        alpha=60000,
        max_iter=50000,
        random_state=42,
    )
    model.coef_ = text_datadict["coef"]
    model.intercept_ = text_datadict["intercept"]
    pred_clip_text = torch.from_numpy(model.predict(voxels[:,0]).reshape(-1, clip_text_seq_dim, clip_emb_dim))
if blurry_recon:
    pred_blurry_vae = torch.zeros((len(images_test), 3136)).to("cpu")
    with open(f'{outdir}/ridge_blurry_weights.pkl', 'rb') as f:
        blurry_datadict = pickle.load(f)
    model = Ridge(
        alpha=60000,
        max_iter=50000,
        random_state=42,
    )
    model.coef_ = blurry_datadict["coef"]
    model.intercept_ = blurry_datadict["intercept"]
    pred_blurry_vae = torch.from_numpy(model.predict(voxels[:,0]).reshape(-1, 3136))
    
    
if normalize_preds:
    for sequence in range(clip_seq_dim):
        std_pred_clip_image = (pred_clip_image[:, sequence] - torch.mean(pred_clip_image[:, sequence],axis=0)) / torch.std(pred_clip_image[:, sequence],axis=0)
        pred_clip_image[:, sequence] = std_pred_clip_image * torch.std(clip_image_train[:, sequence],axis=0) + torch.mean(clip_image_train[:, sequence],axis=0)
    if dual_guidance:
        for sequence in range(clip_text_seq_dim):
            std_pred_clip_text = (pred_clip_text[:, sequence] - torch.mean(pred_clip_text[:, sequence],axis=0)) / torch.std(pred_clip_text[:, sequence],axis=0)
            pred_clip_text[:, sequence] = std_pred_clip_text * torch.std(clip_text_train[:, sequence],axis=0) + torch.mean(clip_text_train[:, sequence],axis=0)
    if blurry_recon:
        std_pred_blurry_vae = (pred_blurry_vae - torch.mean(pred_blurry_vae,axis=0)) / torch.std(pred_blurry_vae,axis=0)
        pred_blurry_vae = std_pred_blurry_vae * torch.std(vae_image_train,axis=0) + torch.mean(vae_image_train,axis=0)


# In[8]:


final_recons = None
final_predcaptions = None
final_clipvoxels = None
final_blurryrecons = None
raw_root = f"/export/raid1/home/kneel027/Second-Sight/output/mental_imagery_paper_b3/{mode}/{model_name}/subject{subj}/"
print("raw_root:", raw_root)
recons_per_sample = 1

for rep in tqdm(range(gen_rep)):
    seed = random.randint(0,10000000)
    utils.seed_everything(seed = seed)
    print(f"seed = {seed}")
    # get all reconstructions    
    # all_images = None
    all_blurryrecons = None
    all_recons = None
    all_predcaptions = []
    all_clipvoxels = None
    
    minibatch_size = 1
    num_samples_per_image = 1
    plotting = False
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for idx in tqdm(range(0,voxels.shape[0]), desc="sample loop"):
            
            clip_voxels = pred_clip_image[idx].unsqueeze(0)
            if dual_guidance:
                clip_text_voxels = pred_clip_text[idx].unsqueeze(0)
            else:
                clip_text_voxels = None
            # Save retrieval submodule outputs
            if all_clipvoxels is None:
                all_clipvoxels = clip_voxels.to('cpu')
            else:
                all_clipvoxels = torch.vstack((all_clipvoxels, clip_voxels.to('cpu')))
            
            if blurry_recon:
                blurred_image = (autoenc.decode(pred_blurry_vae[idx].reshape((1,4,28,28)).half().to(device)/0.18215).sample/ 2 + 0.5).clamp(0,1)
                im = torch.Tensor(blurred_image)
                if all_blurryrecons is None:
                    all_blurryrecons = im.cpu()
                else:
                    all_blurryrecons = torch.vstack((all_blurryrecons, im.cpu()))
                if plotting:
                    plt.figure(figsize=(2,2))
                    plt.imshow(transforms.ToPILImage()(im))
                    plt.axis('off')
                    plt.show()
            
            # Feed outputs through versatile diffusion
            samples_multi = [reconstructor.reconstruct(
                                image=transforms.ToPILImage()(torch.Tensor(blurred_image[0])),
                                c_i=clip_voxels,
                                c_t=clip_text_voxels,
                                n_samples=1,
                                textstrength=0.4,
                                strength=0.85,
                                seed=seed) for _ in range(recons_per_sample)]
            samples = utils.pick_best_recon(samples_multi, clip_voxels, clip_extractor)
            if isinstance(samples, PIL.Image.Image):
                samples = transforms.ToTensor()(samples)
            samples = samples.unsqueeze(0)
            
            if all_recons is None:
                all_recons = samples.cpu()
            else:
                all_recons = torch.vstack((all_recons, samples.cpu()))
            if plotting:
                for s in range(num_samples_per_image):
                    plt.figure(figsize=(2,2))
                    plt.imshow(transforms.ToPILImage()(samples[s]))
                    plt.axis('off')
                    plt.show()
                    
            if plotting: 
                print(model_name)
                err # dont actually want to run the whole thing with plotting=True

            if save_raw:
                # print(f"Saving raw images to {raw_root}/{idx}/{rep}.png")
                os.makedirs(f"{raw_root}/{idx}/", exist_ok=True)
                transforms.ToPILImage()(samples[0]).save(f"{raw_root}/{idx}/{rep}.png")
                transforms.ToPILImage()(all_images[idx]).save(f"{raw_root}/{idx}/ground_truth.png")
                if rep == 0:
                    transforms.ToPILImage()(torch.Tensor(blurred_image[0]).cpu()).save(f"{raw_root}/{idx}/low_level.png")
                    torch.save(clip_voxels, f"{raw_root}/{idx}/clip_image_voxels.pt")
                    if dual_guidance:
                        torch.save(clip_text_voxels, f"{raw_root}/{idx}/clip_text_voxels.pt")
        # resize outputs before saving
        imsize = 256
        # saving
        # print(all_recons.shape)
        # torch.save(all_images,"evals/all_images.pt")
        if final_recons is None:
            final_recons = all_recons.unsqueeze(1)
            # final_predcaptions = np.expand_dims(all_predcaptions, axis=1)
            final_clipvoxels = all_clipvoxels.unsqueeze(1)
            if blurry_recon:
                final_blurryrecons = all_blurryrecons.unsqueeze(1)
        else:
            final_recons = torch.cat((final_recons, all_recons.unsqueeze(1)), dim=1)
            # final_predcaptions = np.concatenate((final_predcaptions, np.expand_dims(all_predcaptions, axis=1)), axis=1)
            final_clipvoxels = torch.cat((final_clipvoxels, all_clipvoxels.unsqueeze(1)), dim=1)
            if blurry_recon:
                final_blurryrecons = torch.cat((all_blurryrecons.unsqueeze(1),final_blurryrecons), dim = 1)
        
if blurry_recon:
    torch.save(final_blurryrecons,f"evals/{model_name}/{model_name}_all_blurryrecons_{mode}.pt")
torch.save(final_recons,f"evals/{model_name}/{model_name}_all_recons_{mode}.pt")
# torch.save(final_predcaptions,f"evals/{model_name}/{model_name}_all_predcaptions_{mode}.pt")
torch.save(final_clipvoxels,f"evals/{model_name}/{model_name}_all_clipvoxels_{mode}.pt")
print(f"saved {model_name} mi outputs!")

# if not utils.is_interactive():
#     sys.exit(0)


# In[ ]:


# imsize = 150
# if all_images.shape[-1] != imsize:
#     all_images = transforms.Resize((imsize,imsize))(transforms.CenterCrop(all_images.shape[2])(all_images)).float()
# if all_recons.shape[-1] != imsize:
#     all_recons = transforms.Resize((imsize,imsize))(transforms.CenterCrop(all_images.shape[2])(all_recons)).float()
# print(all_images.shape, all_recons.shape)
# num_images = all_recons.shape[0]
# num_rows = (2 * num_images + 11) // 12

# # Interleave tensors
# merged = torch.stack([val for pair in zip(all_images, all_recons) for val in pair], dim=0)

# # Calculate grid size
# grid = torch.zeros((num_rows * 12, 3, all_recons.shape[-1], all_recons.shape[-1]))

# # Populate the grid
# grid[:2*num_images] = merged
# grid_images = [transforms.functional.to_pil_image(grid[i]) for i in range(num_rows * 12)]

# # Create the grid image
# grid_image = Image.new('RGB', (all_recons.shape[-1] * 12, all_recons.shape[-1] * num_rows))  # 12 images wide

# # Paste images into the grid
# for i, img in enumerate(grid_images):
#     grid_image.paste(img, (all_recons.shape[-1] * (i % 12), all_recons.shape[-1] * (i // 12)))

# # Create title row image
# title_height = 150
# title_image = Image.new('RGB', (grid_image.width, title_height), color=(255, 255, 255))
# draw = ImageDraw.Draw(title_image)
# font = ImageFont.truetype("arial.ttf", 38)  # Change font size to 3 times bigger (15*3)
# title_text = f"Model: {model_name}, Mode: {mode}"
# bbox = draw.textbbox((0, 0), title_text, font=font)
# text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
# draw.text(((grid_image.width - text_width) / 2, (title_height - text_height) / 2), title_text, fill="black", font=font)

# # Combine title and grid images
# final_image = Image.new('RGB', (grid_image.width, grid_image.height + title_height))
# final_image.paste(title_image, (0, 0))
# final_image.paste(grid_image, (0, title_height))

# final_image.save(f"../figs/{model_name}_{len(all_recons)}recons_{mode}.png")
# print(f"saved ../figs/{model_name}_{len(all_recons)}recons_{mode}.png")


# In[ ]:


if not utils.is_interactive():
    sys.exit(0)

