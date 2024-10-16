#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import contextlib
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator, DeepSpeedPlugin

# from sentence_transformers import SentenceTransformer, util
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
# import evaluate
import pandas as pd

from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from models import GNet8_Encoder

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils

### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

accelerator = Accelerator(split_batches=False, mixed_precision="fp16") # ['no', 'fp8', 'fp16', 'bf16']

print("PID of this process =",os.getpid())
device = accelerator.device
print("device:",device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
num_devices = torch.cuda.device_count()
if num_devices==0 or not distributed: num_devices = 1
num_workers = num_devices
print(accelerator.state)

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)
print = accelerator.print # only print if local_rank=0


# In[2]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "test_pretrained_subj01_40sess_hypatia_pg_sessions40"
    # model_name = "pretest_pretrained_subj01_40sess_hypatia_pg_sessions40"
    mode = "imagery"
    # all_recons_path = f"evals/{model_name}/{model_name}_all_enhancedrecons_{mode}.pt"
    all_recons_path = f"evals/{model_name}/{model_name}_all_recons_{mode}.pt"
    subj = 1
    
    cache_dir = "/weka/proj-medarc/shared/cache"
    data_path = "/weka/proj-medarc/shared/mindeyev2_dataset"
    
    print("model_name:", model_name)

    jupyter_args = f"--model_name={model_name} --subj={subj} --data_path={data_path} --cache_dir={cache_dir} --all_recons_path={all_recons_path} --mode {mode} \
                    --criteria=all"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[39]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--all_recons_path", type=str,
    help="Path to where all_recons.pt is stored",
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
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="Evaluate on which subject?",
)
parser.add_argument(
    "--mode",type=str,default="vision",
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--criteria",type=str, default="all",
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


# # Evals

# In[4]:


if mode == "synthetic":
    all_images = torch.zeros((284, 3, 714, 1360))
    all_images[:220] = torch.load(f"{data_path}/nsddata_stimuli/stimuli/nsdsynthetic/nsd_synthetic_stim_part1.pt")
    #The last 64 stimuli are slightly different for each subject, so we load these separately for each subject
    all_images[220:] = torch.load(f"{data_path}/nsddata_stimuli/stimuli/nsdsynthetic/nsd_synthetic_stim_part2_sub{subj}.pt")
else:
    all_images = torch.load(f"{data_path}/nsddata_stimuli/stimuli/imagery_stimuli_18.pt")


# In[7]:


print("all_recons_path:", all_recons_path)
all_recons_mult = torch.load(all_recons_path).reshape((18, 10, 3, 512, 512))
print("all_recons_mult.shape:", all_recons_mult.shape)

# Residual submodule
try:
    all_clipvoxels_mult = torch.load(f"evals/{model_name}/{model_name}_all_clipvoxels_{mode}.pt").reshape((18, 257, 768))
    print("all_clipvoxels_mult.shape:", all_clipvoxels_mult.shape)
    clip_enabled = True
except:
    clip_enabled = False
# Low-level submodule
if blurry_recon:
    all_blurryrecons_mult = torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons_{mode}.pt")

# model name
model_name_plus_suffix = f"{model_name}_all_recons_{mode}"
print(model_name_plus_suffix)
print(all_images.shape, all_recons_mult.shape)


# In[9]:


# # create full grid of recon comparisons
# from PIL import Image

# imsize = 150
# if all_images.shape[-1] != imsize:
#     all_images = transforms.Resize((imsize,imsize))(transforms.CenterCrop(all_images.shape[2])(all_images)).float()
# if all_recons.shape[-1] != imsize:
#     all_recons = transforms.Resize((imsize,imsize))(transforms.CenterCrop(all_images.shape[2])(all_recons)).float()

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
# grid_image = Image.new('RGB', (all_recons.shape[-1]*12, all_recons.shape[-1] * num_rows))  # 10 images wide

# # Paste images into the grid
# for i, img in enumerate(grid_images):
#     grid_image.paste(img, (all_recons.shape[-1] * (i % 12), all_recons.shape[-1] * (i // 12)))

# grid_image.save(f"../figs/{model_name_plus_suffix}_{len(all_recons)}recons.png")


# In[10]:


# # ground truths, if using NSD-Imagery, we load only the first 12 because the last 6 are conceptual stimuli, for which there was no "ground truth image" to calculate statistics against
# if mode != "synthetic":
#     all_images = all_images[:12]
#     all_recons = all_recons[:12]
#     all_clipvoxels = all_clipvoxels[:12]
#     if blurry_recon:
#         all_blurryrecons = all_blurryrecons[:12]


# In[11]:


# imsize = 256
# if all_images.shape[-1] != imsize:
#     all_images = transforms.Resize((imsize,imsize))(all_images).float()
# if all_recons.shape[-1] != imsize:
#     all_recons = transforms.Resize((imsize,imsize))(all_recons).float()
# if blurry_recon:
#     if all_blurryrecons.shape[-1] != imsize:
#         all_blurryrecons = transforms.Resize((imsize,imsize))(all_blurryrecons).float()
    
# if "enhanced" in model_name_plus_suffix and blurry_recon:
#     print("weighted averaging to improve low-level evals")
#     all_recons = all_recons*.75 + all_blurryrecons*.25


# In[12]:


# # 2 / 117 / 231 / 164 / 619 / 791
# import textwrap
# def wrap_title(title, wrap_width):
#     return "\n".join(textwrap.wrap(title, wrap_width))

# fig, axes = plt.subplots(4,6, figsize=(12,8))
# index = 0
# for j in range(4):
#     for k in range(6):
#         if k%2==0:
#             axes[j][k].imshow(utils.torch_to_Image(all_images[index]))
#             axes[j][k].axis('off')
#         else:
#             axes[j][k].imshow(utils.torch_to_Image(all_recons[index]))
#             axes[j][k].axis('off')
#             index +=1


# # Retrieval eval

# In[13]:


# # Load embedding model
# clip_img_embedder = FrozenOpenCLIPImageEmbedder(
#     arch="ViT-bigG-14",
#     version="laion2b_s39b_b160k",
#     output_tokens=True,
#     only_tokens=True,
# )
# clip_img_embedder.to(device)

# clip_seq_dim = 256
# clip_emb_dim = 1664


# In[14]:


# from scipy import stats

# def get_retrieval_eval(all_images_o, all_clipvoxels_o, plot = False):
#     all_clipvoxels = all_clipvoxels_o.detach().cpu()
#     all_images = all_images_o.detach().cpu()
#     percent_correct_fwds, percent_correct_bwds = [], []
#     percent_correct_fwd, percent_correct_bwd = None, None
    
#     with torch.cuda.amp.autocast(dtype=torch.float16):
#         for test_i, loop in enumerate(tqdm(range(30))):
#             random_samps = np.random.choice(np.arange(len(all_images)), size=4, replace=False)
#             emb = clip_img_embedder.embed_image(all_images[random_samps].to(device)).float() # CLIP-Image
#             emb_ = all_clipvoxels[random_samps].to(device).float() # CLIP-Brain
    
#             # flatten if necessary
#             emb = emb.reshape(len(emb),-1)
#             emb_ = emb_.reshape(len(emb_),-1)
    
#             # l2norm 
#             emb = nn.functional.normalize(emb,dim=-1)
#             emb_ = nn.functional.normalize(emb_,dim=-1)
    
#             labels = torch.arange(len(emb)).to(device)
#             print(emb.shape, emb_.shape)
#             bwd_sim = utils.batchwise_cosine_similarity(emb,emb_)  # clip, brain
#             fwd_sim = utils.batchwise_cosine_similarity(emb_,emb)  # brain, clip
    
#             assert len(bwd_sim) == 4
    
#             percent_correct_fwds = np.append(percent_correct_fwds, utils.topk(fwd_sim, labels,k=1).item())
#             percent_correct_bwds = np.append(percent_correct_bwds, utils.topk(bwd_sim, labels,k=1).item())
    
#             if test_i==0:
#                 print("Loop 0:",percent_correct_fwds, percent_correct_bwds)
                
#     percent_correct_fwd = np.mean(percent_correct_fwds)
#     fwd_sd = np.std(percent_correct_fwds) / np.sqrt(len(percent_correct_fwds))
#     fwd_ci = stats.norm.interval(0.95, loc=percent_correct_fwd, scale=fwd_sd)
    
#     percent_correct_bwd = np.mean(percent_correct_bwds)
#     bwd_sd = np.std(percent_correct_bwds) / np.sqrt(len(percent_correct_bwds))
#     bwd_ci = stats.norm.interval(0.95, loc=percent_correct_bwd, scale=bwd_sd)
    
#     print(f"fwd percent_correct: {percent_correct_fwd:.4f} 95% CI: [{fwd_ci[0]:.4f},{fwd_ci[1]:.4f}]")
#     print(f"bwd percent_correct: {percent_correct_bwd:.4f} 95% CI: [{bwd_ci[0]:.4f},{bwd_ci[1]:.4f}]")
    
#     fwd_sim = np.array(fwd_sim.cpu())
#     bwd_sim = np.array(bwd_sim.cpu())

#     if plot:
#         print("Given Brain embedding, find correct Image embedding")
#         fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(11,12))
#         for trial in range(4):
#             ax[trial, 0].imshow(utils.torch_to_Image(all_images[random_samps][trial]))
#             ax[trial, 0].set_title("original\nimage")
#             ax[trial, 0].axis("off")
#             for attempt in range(3):
#                 which = np.flip(np.argsort(fwd_sim[trial]))[attempt]
#                 ax[trial, attempt+1].imshow(utils.torch_to_Image(all_images[random_samps][which]))
#                 ax[trial, attempt+1].set_title(f"Top {attempt+1}")
#                 ax[trial, attempt+1].axis("off")
#         fig.tight_layout()
#         plt.show()
#     return fwd_sim, bwd_sim, percent_correct_fwd, percent_correct_bwd


# ## 2-way identification

# In[15]:


from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1


# ## PixCorr

# In[16]:


preprocess_pixcorr = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
])

def get_pix_corr(all_images, all_recons):

    
    # Flatten images while keeping the batch dimension
    all_images_flattened = preprocess_pixcorr(all_images).reshape(len(all_images), -1).cpu()
    all_recons_flattened = preprocess_pixcorr(all_recons).view(len(all_recons), -1).cpu()
    
    print(all_images_flattened.shape)
    print(all_recons_flattened.shape)
    
    corrsum = 0
    for i in tqdm(range(len(all_images))):
        corrsum += np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1]
    corrmean = corrsum / len(all_images)
    
    pixcorr = corrmean
    print(pixcorr)
    return pixcorr


# ## SSIM

# In[17]:


# see https://github.com/zijin-gu/meshconv-decoding/issues/3
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity

preprocess_ssim = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
])

def get_ssim(all_images, all_recons):

    
    # convert image to grayscale with rgb2grey
    img_gray = rgb2gray(preprocess_ssim(all_images).permute((0,2,3,1)).cpu())
    recon_gray = rgb2gray(preprocess_ssim(all_recons).permute((0,2,3,1)).cpu())
    print("converted, now calculating ssim...")
    
    ssim_score=[]
    for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
        ssim_score.append(structural_similarity(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))
    
    ssim = np.mean(ssim_score)
    print(ssim)
    return ssim


# ## AlexNet

# In[18]:


from torchvision.models import alexnet, AlexNet_Weights
alex_weights = AlexNet_Weights.IMAGENET1K_V1

alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
alex_model.eval().requires_grad_(False)
preprocess_alexnet = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
def get_alexnet(all_images, all_recons):
    # see alex_weights.transforms()

    
    layer = 'early, AlexNet(2)'
    print(f"\n---{layer}---")
    all_per_correct = two_way_identification(all_recons.to(device).float(), all_images, 
                                                              alex_model, preprocess_alexnet, 'features.4')
    alexnet2 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet2:.4f}")
    
    layer = 'mid, AlexNet(5)'
    print(f"\n---{layer}---")
    all_per_correct = two_way_identification(all_recons.to(device).float(), all_images, 
                                                              alex_model, preprocess_alexnet, 'features.11')
    alexnet5 = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {alexnet5:.4f}")
    return alexnet2, alexnet5


# ## InceptionV3

# In[19]:


from torchvision.models import inception_v3, Inception_V3_Weights
weights = Inception_V3_Weights.DEFAULT
inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                           return_nodes=['avgpool']).to(device)
inception_model.eval().requires_grad_(False)
preprocess_inception = transforms.Compose([
    transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
def get_inceptionv3(all_images, all_recons):
    # see weights.transforms()

    
    all_per_correct = two_way_identification(all_recons.float(), all_images.float(),
                                            inception_model, preprocess_inception, 'avgpool')
            
    inception = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {inception:.4f}")
    return inception


# ## CLIP

# In[20]:


import clip
clip_model, preprocess = clip.load("ViT-L/14", device=device)
preprocess_clip = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

def get_clip(all_images, all_recons):

    
    all_per_correct = two_way_identification(all_recons, all_images,
                                            clip_model.encode_image, preprocess_clip, None) # final layer
    clip_ = np.mean(all_per_correct)
    print(f"2-way Percent Correct: {clip_:.4f}")
    return clip_


# ## Efficient Net

# In[21]:


import scipy as sp
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
weights = EfficientNet_B1_Weights.DEFAULT
eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                    return_nodes=['avgpool'])
eff_model.eval().requires_grad_(False)
preprocess_efficientnet = transforms.Compose([
    transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
def get_efficientnet(all_images, all_recons):
    # see weights.transforms()

    
    gt = eff_model(preprocess_efficientnet(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = eff_model(preprocess_efficientnet(all_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()
    
    effnet = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
    print("Distance:",effnet)
    return effnet


# ## SwAV

# In[22]:


swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
swav_model = create_feature_extractor(swav_model, 
                                    return_nodes=['avgpool'])
swav_model.eval().requires_grad_(False)
preprocess_swav = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
def get_swav(all_images, all_recons):
    gt = swav_model(preprocess_swav(all_images))['avgpool']
    gt = gt.reshape(len(gt),-1).cpu().numpy()
    fake = swav_model(preprocess_swav(all_recons))['avgpool']
    fake = fake.reshape(len(fake),-1).cpu().numpy()
    
    swav = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
    print("Distance:",swav)
    return swav


# # Brain Correlation
# ### Load brain data, brain masks, image lists

# In[41]:


if mode == "synthetic":
    voxels, stimulus = utils.load_nsd_synthetic(subject=subj, average=False, nest=True, data_root=data_path)
else:
    voxels, _ = utils.load_nsd_mental_imagery(subject=subj, mode=mode, stimtype="all", average=True, nest=False, data_root=data_path)
    voxels = voxels[:12]
num_voxels = voxels.shape[-1]
num_test = voxels.shape[0]


# In[42]:


# Load brain region masks
brain_region_masks = {}
with h5py.File(f"{cache_dir}/brain_region_masks.hdf5", "r") as file:
    # Iterate over each subject
    for subject in file.keys():
        subject_group = file[subject]
        # Load the masks data for each subject
        subject_masks = {"nsd_general" : subject_group["nsd_general"][:],
                         "V1" : subject_group["V1"][:], 
                         "V2" : subject_group["V2"][:], 
                         "V3" : subject_group["V3"][:], 
                         "V4" : subject_group["V4"][:],
                         "higher_vis" : subject_group["higher_vis"][:]}
        brain_region_masks[subject] = subject_masks
subject_masks = brain_region_masks[f"subj0{subj}"]


# ### Calculate Brain Correlation scores for each brain area

# In[43]:


from torchmetrics import PearsonCorrCoef
GNet = GNet8_Encoder(device=device,subject=subj,model_path=f"{cache_dir}/gnet_multisubject.pt")

def get_brain_correlation(subject_masks, idx):

    # Prepare image list for input to GNet
    recon_list = []
    for i in range(all_recons.shape[0]):
        img = all_recons[i].detach()
        img = transforms.ToPILImage()(img)
        recon_list.append(img)
        
    PeC = PearsonCorrCoef(num_outputs=len(recon_list))
    beta_primes = GNet.predict(recon_list)
    
    region_brain_correlations = {}
    for region, mask in subject_masks.items():
        score = PeC(voxels[idx,0,mask].unsqueeze(0).moveaxis(0,1), beta_primes[:,mask].moveaxis(0,1))
        region_brain_correlations[region] = float(torch.mean(score))
    print(region_brain_correlations)
    return region_brain_correlations


# In[ ]:


metrics_data = {
    "index_sample": [],
    "PixCorr": [],
    "SSIM": [],
    "AlexNet(2)": [],
    "AlexNet(5)": [],
    "InceptionV3": [],
    "CLIP": [],
    "EffNet-B": [],
    "SwAV": [],
    "FwdRetrieval": [],
    "BwdRetrieval": [],
    "Brain Corr. nsd_general": [],
    "Brain Corr. V1": [],
    "Brain Corr. V2": [],
    "Brain Corr. V3": [],
    "Brain Corr. V4": [],
    "Brain Corr. higher_vis": [],
    "index_image": []  # Add a new column for the index of the image
}

# Iterate over each sample and compute metrics with tqdm and suppressed output
for index_sample in tqdm(range(all_recons_mult.shape[1]), desc="Processing samples"):
    for image_index in range(12):  # Loop over the 12 images
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            all_images_ = all_images[image_index:image_index+1].float()  # Process one image at a time
            all_recons = all_recons_mult[image_index:image_index+1, index_sample, :, :].float()
            if clip_enabled:
                all_clipvoxels = all_clipvoxels_mult[image_index:image_index+1, index_sample, :, :].float()
            # if blurry_recon:
            #     all_blurryrecons = all_blurryrecons_mult[image_index:image_index+1, index_sample, :, :].float()

            # fwd_sim, bwd_sim, percent_correct_fwd, percent_correct_bwd = get_retrieval_eval(all_images_, all_clipvoxels)
            fwd_sim, bwd_sim, percent_correct_fwd, percent_correct_bwd = None, None, None, None
            pixcorr = get_pix_corr(all_images_, all_recons)
            ssim = get_ssim(all_images_, all_recons)
            alexnet2, alexnet5 = get_alexnet(all_images_, all_recons)
            inception = get_inceptionv3(all_images_, all_recons)
            clip_ = get_clip(all_images_, all_recons)
            effnet = get_efficientnet(all_images_, all_recons)
            swav = get_swav(all_images_, all_recons)
            region_brain_correlations = get_brain_correlation(subject_masks, image_index)

        # Append each result to its corresponding list, and store the image index
        metrics_data["index_sample"].append(index_sample)
        metrics_data["PixCorr"].append(pixcorr)
        metrics_data["SSIM"].append(ssim)
        metrics_data["AlexNet(2)"].append(alexnet2)
        metrics_data["AlexNet(5)"].append(alexnet5)
        metrics_data["InceptionV3"].append(inception)
        metrics_data["CLIP"].append(clip_)
        metrics_data["EffNet-B"].append(effnet)
        metrics_data["SwAV"].append(swav)
        metrics_data["FwdRetrieval"].append(percent_correct_fwd)
        metrics_data["BwdRetrieval"].append(percent_correct_bwd)
        metrics_data["Brain Corr. nsd_general"].append(region_brain_correlations["nsd_general"])
        metrics_data["Brain Corr. V1"].append(region_brain_correlations["V1"])
        metrics_data["Brain Corr. V2"].append(region_brain_correlations["V2"])
        metrics_data["Brain Corr. V3"].append(region_brain_correlations["V3"])
        metrics_data["Brain Corr. V4"].append(region_brain_correlations["V4"])
        metrics_data["Brain Corr. higher_vis"].append(region_brain_correlations["higher_vis"])
        metrics_data["index_image"].append(image_index)  # Add image index to the data

# Check that all lists have the same length before creating DataFrame
lengths = [len(values) for values in metrics_data.values()]
if len(set(lengths)) != 1:
    print("Error: Not all metric lists have the same length")
    for metric, values in metrics_data.items():
        print(f"{metric}: {len(values)} items")
else:
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(metrics_data)

    # Rename the index to sample_1, sample_2, etc.
    # df.index = [f'sample_{i+1}' for i in range(df.shape[0])]

    # print(model_name_plus_suffix)
    # print(df.to_string(index=True))

    # Save the table to a CSV file
    os.makedirs('tables/', exist_ok=True)
    df.to_csv(f'tables/{model_name_plus_suffix}.csv', sep='\t')
    
    


# In[45]:


# def get_best_and_medium(df, criteria):
#     if criteria == "all":
#         # Average all metrics
#         scores = df.mean(axis=1)
#     else:
#         # Average the specified criteria
#         scores = df[criteria].mean(axis=1)
    
#     # Get the index of the best score (highest)
#     best_index = scores.idxmax()
    
#     # Get the index of the median score
#     median_index = scores.sort_values().index[len(scores) // 2]
    
#     return best_index, median_index

# # Example usage:
# # criteria = ["AlexNet(2)", "SSIM"]  # or "all"
# best_index, median_index = get_best_and_medium(df, criteria)

# print(f"Best sample: {best_index}")
# print(f"Median sample: {median_index}")


# In[46]:


# # create full grid of recon comparisons
# from PIL import Image

# imsize = 150

# def save_plot(all_images, all_recons, name):
#     if all_images.shape[-1] != imsize:
#         all_images = transforms.Resize((imsize,imsize))(transforms.CenterCrop(all_images.shape[2])(all_images)).float()
#     if all_recons.shape[-1] != imsize:
#         all_recons = transforms.Resize((imsize,imsize))(transforms.CenterCrop(all_images.shape[2])(all_recons)).float()
    
#     num_images = all_recons.shape[0]
#     num_rows = (2 * num_images + 11) // 12
    
#     # Interleave tensors
#     merged = torch.stack([val for pair in zip(all_images, all_recons) for val in pair], dim=0)
    
#     # Calculate grid size
#     grid = torch.zeros((num_rows * 12, 3, all_recons.shape[-1], all_recons.shape[-1]))
    
#     # Populate the grid
#     grid[:2*num_images] = merged
#     grid_images = [transforms.functional.to_pil_image(grid[i]) for i in range(num_rows * 12)]
    
#     # Create the grid image
#     grid_image = Image.new('RGB', (all_recons.shape[-1]*12, all_recons.shape[-1] * num_rows))  # 10 images wide
    
#     # Paste images into the grid
#     for i, img in enumerate(grid_images):
#         grid_image.paste(img, (all_recons.shape[-1] * (i % 12), all_recons.shape[-1] * (i // 12)))
    
#     grid_image.save(f"../figs/{model_name_plus_suffix}_{len(all_recons)}recons_{name}.png")


# In[47]:


# best_idx = int(best_index[-1]) - 1
# median_idx = int(median_index[-1]) - 1

# save_plot(all_images, all_recons_mult[:,best_idx,:,:], "best")
# save_plot(all_images, all_recons_mult[:,median_idx,:,:], "median")


# ### 

# In[ ]:


# # Create a dictionary to store variable names and their corresponding values
# import pandas as pd
# data = {
#     "Metric": ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", "EffNet-B", "SwAV", "FwdRetrieval", "BwdRetrieval",
#                "Brain Corr. nsd_general", "Brain Corr. V1", "Brain Corr. V2", "Brain Corr. V3", "Brain Corr. V4",  "Brain Corr. higher_vis"],
#     "Value": [pixcorr, ssim, alexnet2, alexnet5, inception, clip_, effnet, swav, percent_correct_fwd, percent_correct_bwd, 
#               region_brain_correlations["nsd_general"], region_brain_correlations["V1"], region_brain_correlations["V2"], region_brain_correlations["V3"], region_brain_correlations["V4"], region_brain_correlations["higher_vis"]]}

# df = pd.DataFrame(data)
# print(model_name_plus_suffix)
# print(df.to_string(index=False))
# print(df["Value"].to_string(index=False))

# # save table to txt file
# os.makedirs('tables/',exist_ok=True)
# df["Value"].to_csv(f'tables/{model_name_plus_suffix}.csv', sep='\t', index=False)

