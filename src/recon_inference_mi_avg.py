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


# make output directory
os.makedirs("evals",exist_ok=True)
os.makedirs(f"evals/{model_name}_avg2",exist_ok=True)


# In[4]:


if mode == "synthetic":
    voxels, all_images = utils.load_nsd_synthetic(subject=subj, average=False, nest=True)
elif subj > 8:
    _, _, voxels, all_images = utils.load_imageryrf(subject=subj-8, mode=mode, stimtype="object", average=False, nest=True, split=True)
else:
    voxels, all_images = utils.load_nsd_mental_imagery(subject=subj, mode=mode, stimtype="all", snr=snr, average=False, nest=True)
num_voxels = voxels.shape[-1]


# In[ ]:


clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
clip_seq_dim = 257
clip_emb_dim = 768

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
    layer_mapping = {
        "encoder.mid_block.attentions.0.to_q.weight": "encoder.mid_block.attentions.0.query.weight",
        "encoder.mid_block.attentions.0.to_q.bias": "encoder.mid_block.attentions.0.query.bias",
        "encoder.mid_block.attentions.0.to_k.weight": "encoder.mid_block.attentions.0.key.weight",
        "encoder.mid_block.attentions.0.to_k.bias": "encoder.mid_block.attentions.0.key.bias",
        "encoder.mid_block.attentions.0.to_v.weight": "encoder.mid_block.attentions.0.value.weight",
        "encoder.mid_block.attentions.0.to_v.bias": "encoder.mid_block.attentions.0.value.bias",
        "encoder.mid_block.attentions.0.to_out.0.weight": "encoder.mid_block.attentions.0.proj_attn.weight",
        "encoder.mid_block.attentions.0.to_out.0.bias": "encoder.mid_block.attentions.0.proj_attn.bias",
        "decoder.mid_block.attentions.0.to_q.weight": "decoder.mid_block.attentions.0.query.weight",
        "decoder.mid_block.attentions.0.to_q.bias": "decoder.mid_block.attentions.0.query.bias",
        "decoder.mid_block.attentions.0.to_k.weight": "decoder.mid_block.attentions.0.key.weight",
        "decoder.mid_block.attentions.0.to_k.bias": "decoder.mid_block.attentions.0.key.bias",
        "decoder.mid_block.attentions.0.to_v.weight": "decoder.mid_block.attentions.0.value.weight",
        "decoder.mid_block.attentions.0.to_v.bias": "decoder.mid_block.attentions.0.value.bias",
        "decoder.mid_block.attentions.0.to_out.0.weight": "decoder.mid_block.attentions.0.proj_attn.weight",
        "decoder.mid_block.attentions.0.to_out.0.bias": "decoder.mid_block.attentions.0.proj_attn.bias"
    }

    # Create a new state dictionary with the renamed layers
    new_ckpt = {}
    for old_key, value in ckpt.items():
        new_key = layer_mapping.get(old_key, old_key)  # Get the new key, or use the old key if not in mapping
        new_ckpt[new_key] = value
    autoenc.load_state_dict(new_ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)
    
class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x
        
model = MindEyeModule()

class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_sizes, out_features, seq_len): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = torch.cat([self.linears[subj_idx](x[:,seq]).unsqueeze(1) for seq in range(seq_len)], dim=1)
        return out
        
model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim, seq_len=seq_len)

from diffusers.models.vae import Decoder
from models import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=seq_len, n_blocks=n_blocks,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim, 
                          blurry_recon=blurry_recon, text_clip=dual_guidance) 
utils.count_params(model.ridge)
utils.count_params(model.backbone)
utils.count_params(model)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 64
heads = clip_emb_dim//64 # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
        learned_query_mode="pos_emb"
    )
model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
if dual_guidance:
    prior_network_txt = PriorNetwork_txt(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = 77,
            learned_query_mode="pos_emb"
        )


    model.diffusion_prior_txt = BrainDiffusionPrior(
        net=prior_network_txt,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
model.to(device)

utils.count_params(model.diffusion_prior)
if dual_guidance:
    utils.count_params(model.diffusion_prior_txt)
utils.count_params(model)

# Load pretrained model ckpt
tag='last'
outdir = os.path.abspath(f'../train_logs/{model_name}')
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
try:
    checkpoint = torch.load(outdir+f'/{tag}.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    layer_mapping = {
        "backbone.bupsampler.mid_block.attentions.0.to_q.weight": "backbone.bupsampler.mid_block.attentions.0.query.weight",
        "backbone.bupsampler.mid_block.attentions.0.to_q.bias": "backbone.bupsampler.mid_block.attentions.0.query.bias",
        "backbone.bupsampler.mid_block.attentions.0.to_k.weight": "backbone.bupsampler.mid_block.attentions.0.key.weight",
        "backbone.bupsampler.mid_block.attentions.0.to_k.bias": "backbone.bupsampler.mid_block.attentions.0.key.bias",
        "backbone.bupsampler.mid_block.attentions.0.to_v.weight": "backbone.bupsampler.mid_block.attentions.0.value.weight",
        "backbone.bupsampler.mid_block.attentions.0.to_v.bias": "backbone.bupsampler.mid_block.attentions.0.value.bias",
        "backbone.bupsampler.mid_block.attentions.0.to_out.0.weight": "backbone.bupsampler.mid_block.attentions.0.proj_attn.weight",
        "backbone.bupsampler.mid_block.attentions.0.to_out.0.bias": "backbone.bupsampler.mid_block.attentions.0.proj_attn.bias"
    }
    new_ckpt = {}
    for old_key, value in state_dict.items():
        new_key = layer_mapping.get(old_key, old_key)  # Get the new key, or use the old key if not in mapping
        new_ckpt[new_key] = value
    
    model.load_state_dict(new_ckpt, strict=True)
    del checkpoint
except: # probably ckpt is saved using deepspeed format
    import deepspeed
    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
print("ckpt loaded!")


# In[7]:


print('Creating versatile diffusion reconstruction pipeline...')
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel
# vd_cache_dir = "/home/naxos2-raid25/kneel027/home/kneel027/fMRI-reconstruction-NSD/versatile_diffusion"
# try:
#     vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(cache_dir).to(device)
# except:
print("Downloading Versatile Diffusion to", cache_dir)
vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(
        "shi-labs/versatile-diffusion",
        torch_dtype=torch.float16,
        cache_dir = cache_dir).to(device)
vd_pipe.remove_unused_weights()
vd_pipe.image_unet.eval()
vd_pipe.vae.eval()
vd_pipe.image_unet.requires_grad_(False)
vd_pipe.vae.requires_grad_(False)

vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(cache_dir + "/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7", subfolder="scheduler")
num_inference_steps = 20

# Set weighting of Dual-Guidance 
if dual_guidance:
    text_image_ratio = .4 # .5 means equally weight text and image, 0 means use only image
else:
    text_image_ratio = 0.
for name, module in vd_pipe.image_unet.named_modules():
    if isinstance(module, DualTransformer2DModel):
        module.mix_ratio = text_image_ratio
        for i, type in enumerate(("text", "image")):
            if type == "text":
                module.condition_lengths[i] = 77
                module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
            else:
                module.condition_lengths[i] = 257
                module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

unet = vd_pipe.image_unet
vae = vd_pipe.vae
noise_scheduler = vd_pipe.scheduler


# In[8]:


final_recons = None
final_predcaptions = None
final_clipvoxels = None
final_blurryrecons = None

model_name = model_name + "_avg2"

recons_per_sample = 16

for rep in tqdm(range(gen_rep)):
    utils.seed_everything(seed = random.randint(0,10000000))
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
            voxel = voxels[idx]
            voxel = voxel.to(device)
            voxel = torch.mean(voxel, dim=0)
            # for trial_rep in range(voxel.shape[0]):
            voxel_ridge = model.ridge(voxel[None,None],0) # 0th index of subj_list
            backbone, backbone_txt, clip_voxels, blurry_image_enc = model.backbone(voxel_ridge)
            blurry_image_enc = blurry_image_enc[0]
            # Save retrieval submodule outputs
            if all_clipvoxels is None:
                all_clipvoxels = clip_voxels.to('cpu')
            else:
                all_clipvoxels = torch.vstack((all_clipvoxels, clip_voxels.to('cpu')))
            
            # Feed voxels through versatile diffusion diffusion prior
            backbone = backbone.repeat(recons_per_sample, 1, 1)
            prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, 
                            text_cond = dict(text_embed = backbone), 
                            cond_scale = 1., timesteps = 20)
            prior_out_txt = None
            if dual_guidance:
                backbone_txt = backbone_txt.repeat(recons_per_sample, 1, 1)
                prior_out_txt = model.diffusion_prior_txt.p_sample_loop(backbone_txt.shape, 
                                text_cond = dict(text_embed = backbone_txt), 
                                cond_scale = 1., timesteps = 20)
            # pred_caption_emb = clip_convert(prior_out)
            # generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
            # generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
            # all_predcaptions = np.hstack((all_predcaptions, generated_caption))
            
            if blurry_recon:
                blurred_image = (autoenc.decode(blurry_image_enc/0.18215).sample/ 2 + 0.5).clamp(0,1)
                
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
            
            # Feed diffusion prior outputs through versatile diffusion
            generator = torch.Generator(device=device)
            samples, brain_recons, best_picks = utils.versatile_diffusion_recon(brain_clip_embeddings=prior_out, 
                                proj_embeddings = clip_voxels, 
                                img_lowlevel = blurred_image, 
                                img2img_strength = .85, 
                                text_token=prior_out_txt,
                                clip_extractor = clip_extractor, 
                                vae=vae, 
                                unet=unet, 
                                noise_scheduler=noise_scheduler, 
                                generator=generator,
                                num_inference_steps = num_inference_steps,
                                recons_per_sample=recons_per_sample)
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
torch.save(final_clipvoxels,f"evals/{model_name}/{model_name}all_clipvoxels_{mode}.pt")
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

