#!/usr/bin/env python
# coding: utf-8

# # Import packages & functions

# In[ ]:


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
from accelerate import Accelerator
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from versatile_diffusion import Reconstructor
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True
from sklearn.linear_model import Ridge
import pickle
# custom functions #
import utils


# In[ ]:


### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

data_type = torch.float16 # change depending on your mixed_precision
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1

# First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!
accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
if utils.is_interactive(): # set batch size here if using interactive notebook instead of submitting job
    global_batch_size = batch_size = 8
else:
    global_batch_size = os.environ["GLOBAL_BATCH_SIZE"]
    batch_size = int(os.environ["GLOBAL_BATCH_SIZE"]) // num_devices


# In[ ]:


print("PID of this process =",os.getpid())
device = accelerator.device
print("device:",device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
num_devices = torch.cuda.device_count()
if num_devices==0 or not distributed: num_devices = 1
num_workers = num_devices
print(accelerator.state)

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
print = accelerator.print # only print if local_rank=0


# # Configurations

# In[ ]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "testing"
    print("model_name:", model_name)
    
    # global_batch_size and batch_size should already be defined in the 2nd cell block
    jupyter_args = f"--data_path=../dataset/ \
                    --cache_dir=../cache/ \
                    --model_name={model_name} \
                    --batch_size=64 \
                    --no-multi_subject --subj=1 --num_sessions=40 \
                    --hidden_dim=1024 --clip_scale=1. \
                    --no-blurry_recon --blur_scale=.5  \
                    --seq_past=0 --seq_future=0 \
                    --no-use_prior --prior_scale=30 \
                    --n_blocks=4 --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --no-use_image_aug \
                    --ckpt_interval=1 --ckpt_saving"

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
    "--weight_decay",type=float,default=1e-2,
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
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir,exist_ok=True)
    
if use_image_aug or blurry_recon:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
if use_image_aug:
    img_augment = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
        same_on_batch=False,
        data_keys=["input"],
    )
    
if multi_subject:
    if train_imageryrf:
            # 9,10,11 is ImageryRF subjects
        if no_nsd:
            subj_list = np.arange(9,12)
        else:
            subj_list = np.arange(1,12)
    else:
        subj_list = np.arange(1,9)
    subj_list = subj_list[subj_list != subj]
else:
    subj_list = [subj]

print("subj_list", subj_list, "num_sessions", num_sessions)


# # Prep data, models, and dataloaders

# ### Creating wds dataloader, preload betas and all 73k possible images

# In[ ]:


def my_split_by_node(urls): return urls
num_voxels_list = []
num_devices = 1
if multi_subject:
    nsessions_allsubj=np.array([40, 40, 32, 30, 40, 32, 40, 30])
    num_samples_per_epoch = (750*40) // num_devices 
else:
    num_samples_per_epoch = (750*num_sessions) // num_devices 

print("dividing batch size by subj_list, which will then be concatenated across subj during training...") 
batch_size = batch_size // len(subj_list)

num_iterations_per_epoch = num_samples_per_epoch // (batch_size*len(subj_list))

print("batch_size =", batch_size, "num_iterations_per_epoch =",num_iterations_per_epoch, "num_samples_per_epoch =",num_samples_per_epoch)



# In[ ]:


train_data = {}
train_dl = {}
num_voxels = {}
voxels = {}
for s in subj_list:
    print(f"Training with {num_sessions} sessions")
    # If an NSD subject
    if s < 9:
        if multi_subject:
            train_url = f"{data_path}/wds/subj{s:02d}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
        else:
            train_url = f"{data_path}/wds/subj{s:02d}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
        print(train_url)
        
        train_data[f'subj{s:02d}'] = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                            .shuffle(750, initial=1500, rng=random.Random(42))\
                            .decode("torch")\
                            .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                            .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
        train_dl[f'subj{s:02d}'] = torch.utils.data.DataLoader(train_data[f'subj{s:02d}'], batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
        # betas = utils.create_snr_betas(subject=s, data_type=data_type, data_path=data_path, threshold = snr_threshold)
        betas = utils.load_subject_based_on_rank_order_rois(excluded_subject=s, data_type=data_type, top_n_rois=top_n_rank_order_rois, samplewise=samplewise_rank_order_rois)
        x_train, valid_nsd_ids_train, x_test, test_nsd_ids = utils.load_nsd(subject=s, betas=betas, data_path=data_path)
        print(x_train.shape, valid_nsd_ids_train.shape)
        num_voxels_list.append(x_train[0].shape[-1])
        num_voxels[f'subj{s:02d}'] = x_train[0].shape[-1]
        voxels[f'subj{s:02d}'] = x_train
    elif s < 12:
        train_url = ""
        test_url = ""
        betas, images, _, _ = utils.load_imageryrf(subject=int(s-8), mode=mode, mask=True, stimtype="object", average=False, nest=False, split=True)
        betas = torch.where(torch.isnan(betas), torch.zeros_like(betas), betas)
        betas = betas.to("cpu").to(data_type)
        num_voxels_list.append(betas[0].shape[-1])
        num_voxels[f'subj{s:02d}'] = betas[0].shape[-1]
        num_nan_values = torch.sum(torch.isnan(betas))
        print("Number of NaN values in betas:", num_nan_values.item())
        indices = torch.randperm(len(betas))
        shuffled_betas = betas[indices]
        shuffled_images = images[indices]
        train_data[f'subj{s:02d}'] = torch.utils.data.TensorDataset(shuffled_betas, shuffled_images)
        train_dl[f'subj{s:02d}'] = torch.utils.data.DataLoader(train_data[f'subj{s:02d}'], batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
        
        
    # elif s < 15:
    #     betas, images = utils.load_imageryrf(subject=int(s-11), mode="imagery", mask=True, stimtype="object", average=False, nest=False)
    #     betas = torch.where(torch.isnan(betas), torch.zeros_like(betas), betas)
    #     betas = betas.to("cpu").to(data_type)
    #     num_voxels_list.append(betas[0].shape[-1])
    #     num_voxels[f'subj{s:02d}'] = betas[0].shape[-1]
        
    #     indices = torch.randperm(len(betas))
    #     shuffled_betas = betas[indices]
    #     shuffled_images = images[indices]
    #     train_data[f'subj{s:02d}'] = torch.utils.data.TensorDataset(shuffled_betas, shuffled_images)
    #     train_dl[f'subj{s:02d}'] = torch.utils.data.DataLoader(train_data[f'subj{s:02d}'], batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
    print(f"num_voxels for subj{s:02d}: {num_voxels[f'subj{s:02d}']}")

print("Loaded all subj train dls and betas!\n")

# Validate only on one subject (doesn't support ImageryRF)
if multi_subject: 
    subj = subj_list[0] # cant validate on the actual held out person so picking first in subj_list
if not new_test: # using old test set from before full dataset released (used in original MindEye paper)
    if subj==3:
        num_test=2113
    elif subj==4:
        num_test=1985
    elif subj==6:
        num_test=2113
    elif subj==8:
        num_test=1985
    else:
        num_test=2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
elif new_test: # using larger test set from after full dataset released
    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"
print(test_url)
if subj < 9:
    test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=1500, rng=random.Random(42))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
else:
    _, _, betas, images = utils.load_imageryrf(subject=int(subj-8), mode=mode, mask=True, stimtype="object", average=False, nest=True, split=True)
    num_test = len(betas)
    betas = torch.where(torch.isnan(betas), torch.zeros_like(betas), betas)
    betas = betas.to("cpu").to(data_type)
    num_nan_values = torch.sum(torch.isnan(betas))
    print("Number of NaN values in test betas:", num_nan_values.item())
    test_data = torch.utils.data.TensorDataset(betas, images)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")

seq_len = seq_past + 1 + seq_future
print(f"currently using {seq_len} seq_len (chose {seq_past} past behav and {seq_future} future behav)")


# In[ ]:


# Load 73k NSD images
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images'] # if you go OOM you can remove the [:] so it isnt preloaded to cpu! (will require a few edits elsewhere tho)
# images = torch.Tensor(images).to("cpu").to(data_type)
print("Loaded all 73k possible NSD images to cpu!", images.shape)

# Load 73k NSD captions
captions = np.load(f'{data_path}/preprocessed_data/annots_73k.npy')
print("Loaded all 73k NSD captions to cpu!", captions.shape)


# ## Load models

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
    # Create a mapping from the old layer names to the new layer names
    autoenc.load_state_dict(ckpt)
    
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)


# ### VD/CLIP image embeddings  model

# In[ ]:


clip_emb_dim = 768
clip_seq_dim = 257
clip_text_seq_dim=77
clip_extractor = Reconstructor(device=device, cache_dir=cache_dir)
clip_variant = "ViT-L-14"
text_embedding_variant = "ViT-L-14"
image_embedding_variant = "ViT-L-14"
clip_text_emb_dim=768



# # Creating block of CLIP embeddings

# In[ ]:


file_path = f"{data_path}/preprocessed_data/{image_embedding_variant}_image_embeddings.pt"
emb_batch_size = 50
if not os.path.exists(file_path):
    # Generate CLIP Image embeddings
    print("Generating CLIP Image embeddings!")
    clip_image = torch.zeros((len(images), clip_seq_dim, clip_emb_dim)).to("cpu")
    for i in tqdm(range(len(images) // emb_batch_size), desc="Encoding images..."):
        batch_images = images[i * emb_batch_size:i * emb_batch_size + emb_batch_size]
        batch_embeddings = clip_extractor.embed_image(torch.from_numpy(batch_images)).detach().to("cpu")
        clip_image[i * emb_batch_size:i * emb_batch_size + emb_batch_size] = batch_embeddings

    torch.save(clip_image, file_path)
else:
    clip_image = torch.load(file_path)
clip_image_train = torch.zeros((len(valid_nsd_ids_train), clip_seq_dim, clip_emb_dim)).to("cpu")
for i, idx in enumerate(valid_nsd_ids_train):
    clip_image_train[i] = clip_image[idx].reshape((-1, clip_seq_dim, clip_emb_dim))
file_path = f"{data_path}/preprocessed_data/subject{subj}/{image_embedding_variant}_image_embeddings_train.pt"
if not os.path.exists(file_path):
    torch.save(clip_image_train, file_path)
        
if dual_guidance:
    file_path_txt = f"{data_path}/preprocessed_data/{text_embedding_variant}_text_embeddings.pt"
    if not os.path.exists(file_path_txt):
        # Generate CLIP Text embeddings
        print("Generating CLIP Text embeddings!")
        clip_text = torch.zeros((len(captions), clip_text_seq_dim, clip_text_emb_dim)).to("cpu")
        for i in tqdm(range(len(captions) // emb_batch_size), desc="Encoding captions..."):
            batch_captions = captions[i * emb_batch_size:i * emb_batch_size + emb_batch_size].tolist()
            clip_text[i * emb_batch_size:i * emb_batch_size + emb_batch_size] = clip_extractor.embed_text(batch_captions).detach().to("cpu")
        torch.save(clip_text, file_path_txt)
    else:
        clip_text = torch.load(file_path_txt)
    clip_text_train = torch.zeros((len(valid_nsd_ids_train), clip_text_seq_dim, clip_text_emb_dim)).to("cpu")
    for i, idx in enumerate(valid_nsd_ids_train):
        clip_text_train[i] = clip_text[idx].reshape((-1, clip_text_seq_dim, clip_text_emb_dim))
    file_path_txt = f"{data_path}/preprocessed_data/subject{subj}/{text_embedding_variant}_text_embeddings_train.pt"
    if not os.path.exists(file_path_txt):
        torch.save(clip_text_train, file_path_txt)
        

if blurry_recon:
    file_path = f"{data_path}/preprocessed_data/autoenc_image_embeddings.pt"
    if not os.path.exists(file_path):
        # Generate CLIP Image embeddings
        print("Generating VAE Image embeddings!")
        vae_image = torch.zeros((len(images), 3136)).to("cpu")
        with torch.cuda.amp.autocast(dtype=torch.float16):

            for i in tqdm(range(len(images) // emb_batch_size), desc="Encoding images..."):
                batch_images = images[i * emb_batch_size:i * emb_batch_size + emb_batch_size]
                batch_images = 2 * torch.from_numpy(batch_images).unsqueeze(0).detach().to(device=device, dtype=torch.float16) - 1
                batch_embeddings = (autoenc.encode(batch_images).latent_dist.mode() * 0.18215).detach().to("cpu").reshape(emb_batch_size, -1)
                vae_image[i * emb_batch_size:i * emb_batch_size + emb_batch_size] = batch_embeddings


    else:
        vae_image = torch.load(file_path)
    vae_image_train = torch.zeros((len(valid_nsd_ids_train), 3136)).to("cpu")
    for i, idx in enumerate(valid_nsd_ids_train):
        vae_image_train[i] = vae_image[idx]
    file_path_vae = f"{data_path}/preprocessed_data/subject{subj}/autoenc_image_embeddings_train.pt"
    if not os.path.exists(file_path_vae):
        torch.save(vae_image_train, file_path_vae)
print(f"Loaded train image clip and text clip for subj{subj}!", clip_image_train.shape)


# In[ ]:


# Filter to only ones needed during trainin

clip_image_train = torch.zeros((len(valid_nsd_ids_train), clip_seq_dim, clip_emb_dim)).to("cpu")
clip_text_train = torch.zeros((len(valid_nsd_ids_train), clip_text_seq_dim, clip_emb_dim)).to("cpu")
vae_image_train = torch.zeros((len(valid_nsd_ids_train), 3136)).to("cpu")
for i, idx in enumerate(valid_nsd_ids_train):
    clip_image_train[i] =  clip_image[idx].reshape((-1, clip_seq_dim, clip_emb_dim))
    clip_text_train[i] = clip_text[idx].reshape((-1, clip_text_seq_dim, clip_emb_dim))
    vae_image_train[i] = vae_image[idx].reshape((-1, 3136))
print(f"Loaded train image clip and text clip for subj{subj}!", clip_image_train.shape)


# # Train Ridge regression models

# In[ ]:


start = time.time()
ridge_weights = np.zeros((clip_seq_dim * clip_emb_dim, num_voxels[f'subj{s:02d}'])).astype(np.float32)
ridge_biases = np.zeros((clip_seq_dim * clip_emb_dim)).astype(np.float32)
print(f"Training Ridge CLIP Image model with alpha={weight_decay}")
model = Ridge(
    alpha=weight_decay,
    max_iter=50000,
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
    ridge_weights_txt = np.zeros((clip_text_seq_dim * clip_emb_dim, num_voxels[f'subj{s:02d}'])).astype(np.float32)
    ridge_biases_txt = np.zeros((clip_text_seq_dim * clip_emb_dim)).astype(np.float32)
    print(f"Training Ridge CLIP Text model with alpha={weight_decay}")
    model = Ridge(
        alpha=weight_decay,
        max_iter=50000,
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
    ridge_weights_blurry = np.zeros((3136,num_voxels[f'subj{s:02d}'])).astype(np.float32)
    ridge_biases_blurry = np.zeros((3136,)).astype(np.float32)
    print(f"Training Ridge Blurry recon model with alpha={weight_decay}")
    model = Ridge(
        alpha=weight_decay,
        max_iter=50000,
        random_state=42,
    )
    model.fit(x_train, vae_image_train)
    ridge_weights_blurry = model.coef_
    ridge_biases_blurry = model.intercept_
    datadict = {"coef" : ridge_weights_blurry, "intercept" : ridge_biases_blurry}
    # Save the regression weights
    with open(f'{outdir}/ridge_blurry_weights.pkl', 'wb') as f:
        pickle.dump(datadict, f)


print(f"Elapsed training time for {model_name}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

