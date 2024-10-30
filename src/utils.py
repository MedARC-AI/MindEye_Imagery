import numpy as np
import pandas as pd
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import random
import os
import pickle
from scipy.io import loadmat
import matplotlib.pyplot as plt
import math
import webdataset as wds
from tqdm import tqdm
import nibabel as nb
import os.path as op

import json
from PIL import Image
import requests
import time 
import h5py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x

def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]

def batchwise_pearson_correlation(Z, B):
    # Calculate means
    Z_mean = torch.mean(Z, dim=1, keepdim=True)
    B_mean = torch.mean(B, dim=1, keepdim=True)

    # Subtract means
    Z_centered = Z - Z_mean
    B_centered = B - B_mean

    # Calculate Pearson correlation coefficient
    numerator = Z_centered @ B_centered.T
    Z_centered_norm = torch.linalg.norm(Z_centered, dim=1, keepdim=True)
    B_centered_norm = torch.linalg.norm(B_centered, dim=1, keepdim=True)
    denominator = Z_centered_norm @ B_centered_norm.T

    pearson_correlation = (numerator / denominator)
    return pearson_correlation

def batchwise_cosine_similarity(Z,B):
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def prenormed_batchwise_cosine_similarity(Z,B):
    return (Z @ B.T).T

def cosine_similarity(Z,B,l=0):
    Z = nn.functional.normalize(Z, p=2, dim=1)
    B = nn.functional.normalize(B, p=2, dim=1)
    # if l>0, use distribution normalization
    # https://twitter.com/YifeiZhou02/status/1716513495087472880
    Z = Z - l * torch.mean(Z,dim=0)
    B = B - l * torch.mean(B,dim=0)
    cosine_similarity = (Z @ B.T).T
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def get_non_diagonals(a):
    a = torch.triu(a,diagonal=1)+torch.tril(a,diagonal=-1)
    # make diagonals -1
    a=a.fill_diagonal_(-1)
    return a

def gather_features(image_features, voxel_features, accelerator):  
    all_image_features = accelerator.gather(image_features.contiguous())
    if voxel_features is not None:
        all_voxel_features = accelerator.gather(voxel_features.contiguous())
        return all_image_features, all_voxel_features
    return all_image_features

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def soft_siglip_loss(preds, targs, temp, bias):
    temp = torch.exp(temp)
    
    logits = (preds @ targs.T) * temp + bias
    # diagonals (aka paired samples) should be >0 and off-diagonals <0
    labels = (targs @ targs.T) - 1 + (torch.eye(len(targs)).to(targs.dtype).to(targs.device))

    loss1 = -torch.sum(nn.functional.logsigmoid(logits * labels[:len(preds)])) / len(preds)
    loss2 = -torch.sum(nn.functional.logsigmoid(logits.T * labels[:,:len(preds)])) / len(preds)
    loss = (loss1 + loss2)/2
    return loss

def mixco_hard_siglip_loss(preds, targs, temp, bias, perm, betas):
    temp = torch.exp(temp)
    
    probs = torch.diag(betas)
    probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

    logits = (preds @ targs.T) * temp + bias
    labels = probs * 2 - 1
    #labels = torch.eye(len(targs)).to(targs.dtype).to(targs.device) * 2 - 1
    
    loss1 = -torch.sum(nn.functional.logsigmoid(logits * labels)) / len(preds)
    loss2 = -torch.sum(nn.functional.logsigmoid(logits.T * labels)) / len(preds)
    loss = (loss1 + loss2)/2
    return loss

def mixco(voxels, beta=0.15, s_thresh=0.5, perm=None, betas=None, select=None):
    if perm is None:
        perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    if betas is None:
        betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    if select is None:
        select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def mixco_clip_target(clip_target, perm, select, betas):
    clip_target_shuffle = clip_target[perm]
    clip_target[select] = clip_target[select] * betas[select].reshape(-1, 1) + \
        clip_target_shuffle[select] * (1 - betas[select]).reshape(-1, 1)
    return clip_target

def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss
    
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'param counts:\n{total:,} total\n{trainable:,} trainable')
    return trainable
    
def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def resize(img, img_size=128):
    if img.ndim == 3: img = img[None]
    return nn.functional.interpolate(img, size=(img_size, img_size), mode='nearest')

pixcorr_preprocess = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
])
def pixcorr(images,brains,nan=True):
    all_images_flattened = pixcorr_preprocess(images).reshape(len(images), -1)
    all_brain_recons_flattened = pixcorr_preprocess(brains).view(len(brains), -1)
    if nan:
        corrmean = torch.nanmean(torch.diag(batchwise_pearson_correlation(all_images_flattened, all_brain_recons_flattened)))
    else:
        corrmean = torch.mean(torch.diag(batchwise_pearson_correlation(all_images_flattened, all_brain_recons_flattened)))
    return corrmean

def select_annotations(annots, random=True):
    """
    There are 5 annotations per image. Select one of them for each image.
    """
    for i, b in enumerate(annots):
        t = ''
        if random:
            # select random non-empty annotation
            while t == '':
                rand = torch.randint(5, (1,1))[0][0]
                t = b[rand]
        else:
            # select first non-empty annotation
            for j in range(5):
                if b[j] != '':
                    t = b[j]
                    break
        if i == 0:
            txt = np.array(t)
        else:
            txt = np.vstack((txt, t))
    txt = txt.flatten()
    return txt

from generative_models.sgm.util import append_dims

def unclip_recon(x, diffusion_engine, vector_suffix,
                 num_samples=1, offset_noise_level=0.04):
    assert x.ndim==3
    if x.shape[0]==1:
        x = x[[0]]
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), diffusion_engine.ema_scope():
        z = torch.randn(num_samples,4,96,96).to(device) # starting noise, can change to VAE outputs of initial image for img2img

        # clip_img_tokenized = clip_img_embedder(image) 
        # tokens = clip_img_tokenized
        token_shape = x.shape
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        tokens = torch.randn_like(x)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)
        # samples = torch.clamp((samples_x + .5) / 2.0, min=0.0, max=1.0)
        return samples
    
def prepare_low_level_latents(img_lowlevel, 
                              img2img_strength,
                              vae, 
                              noise_scheduler, 
                              generator,
                              num_inference_steps,
                              recons_per_sample=16):
    # 5b. Prepare latent variables
    normalize = transforms.Normalize(np.array([0.48145466, 0.4578275, 0.40821073]), np.array([0.26862954, 0.26130258, 0.27577711]))
    # use img_lowlevel for img2img initialization
    img_lowlevel = transforms.Resize((512, 512))(img_lowlevel)
    # img_lowlevel = normalize(img_lowlevel)
    init_latents = vae.encode(img_lowlevel.to(device).to(vae.dtype)).latent_dist.sample(generator)
    init_latents = vae.config.scaling_factor * init_latents
    init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)
    
    init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = noise_scheduler.timesteps[t_start:]
    latent_timestep = timesteps[:1].repeat(recons_per_sample)

    noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                        generator=generator, dtype=init_latents.dtype)
    latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep.int())
    return latents
    
def versatile_diffusion_recon(brain_clip_embeddings, 
                              proj_embeddings, 
                              img_lowlevel, 
                              text_token,
                              img2img_strength, 
                              clip_extractor, 
                              vae, 
                              unet, 
                              noise_scheduler, 
                              generator,
                              num_inference_steps,
                              recons_per_sample=16,
                              guidance_scale = 3.5,
                              seed=42):
    for samp in range(len(brain_clip_embeddings)):
        brain_clip_embeddings[samp] = brain_clip_embeddings[samp]/(brain_clip_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
        
    input_embedding = brain_clip_embeddings
    if text_token is not None:
        prompt_embeds = text_token
        # prompt_embeds = text_token.repeat(recons_per_sample, 1, 1)
        # for samp in range(len(prompt_embeds)):
        #     prompt_embeds[samp] = prompt_embeds[samp]/(prompt_embeds[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
    else:
        prompt_embeds = torch.zeros(len(input_embedding),77,768)
    
    if unet is not None:
        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = unet.config.sample_size * vae_scale_factor
        width = unet.config.sample_size * vae_scale_factor
    
    if do_classifier_free_guidance:
        input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
        prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype)
    
    # dual_prompt_embeddings
    # print(prompt_embeds.shape)
    # print(input_embedding.shape)
    input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)
    # 4. Prepare timesteps
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

    # 5b. Prepare latent variables
    batch_size = input_embedding.shape[0] // 2 # divide by 2 bc we doubled it for classifier-free guidance
    shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
    if img_lowlevel is not None: # use img_lowlevel for img2img initialization
        img_lowlevel = torch.nn.functional.interpolate(img_lowlevel, size=(512, 512), mode='bilinear', align_corners=False)
        init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = noise_scheduler.timesteps[t_start:]
        latent_timestep = timesteps[:1].repeat(batch_size)
        
        img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
        init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
        init_latents = vae.config.scaling_factor * init_latents
        init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

        noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                            generator=generator, dtype=input_embedding.dtype)
        init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
        latents = init_latents
    else:
        timesteps = noise_scheduler.timesteps
        latents = torch.randn([recons_per_sample, 4, 64, 64], device=device,
                                generator=generator, dtype=input_embedding.dtype)
        latents = latents * noise_scheduler.init_noise_sigma
    # 7. Denoising loop
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t).to(device)
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    recons = decode_latents(latents,vae).detach().cpu()
    
    brain_recons = recons.unsqueeze(0)
    
    # pick best reconstruction out of several
    best_picks = np.zeros(1).astype(np.int16)

    v2c_reference_out = nn.functional.normalize(proj_embeddings.view(len(proj_embeddings),-1),dim=-1)
    sims=[]
    for im in range(recons_per_sample): 
        currecon = clip_extractor.embed_image(brain_recons[0,[im]].float()).to(proj_embeddings.device).to(proj_embeddings.dtype)
        currecon = nn.functional.normalize(currecon.view(len(currecon),-1),dim=-1)
        cursim = batchwise_cosine_similarity(v2c_reference_out,currecon)
        sims.append(cursim.item())
    best_picks[0] = int(np.nanargmax(sims))  
     
    recon_img = brain_recons[:, best_picks[0]]
    
    return recon_img, brain_recons, best_picks

def pick_best_recon(brain_recons, proj_embeddings, clip_extractor, hidden=False):
    # pick best reconstruction out of several
    best_picks = np.zeros(1).astype(np.int16)
    v2c_reference_out = nn.functional.normalize(proj_embeddings.view(len(proj_embeddings),-1),dim=-1)
    sims=[]
    for im in range(len(brain_recons)): 
        currecon = clip_extractor.embed_image(brain_recons[im], hidden=hidden).to(proj_embeddings.device).to(proj_embeddings.dtype)
        currecon = nn.functional.normalize(currecon.view(len(currecon),-1),dim=-1)
        cursim = batchwise_cosine_similarity(v2c_reference_out,currecon)
        sims.append(cursim.item())
    best_picks[0] = int(np.nanargmax(sims))  
     
    recon_img = brain_recons[best_picks[0]]
    
    return recon_img

def decode_latents(latents,vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

#  Numpy Utility 
def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 
        
# Torch fwRF
def get_value(_x):
    return np.copy(_x.data.cpu().numpy())

def soft_cont_loss(student_preds, teacher_preds, teacher_aug_preds, temp=0.125):
    teacher_teacher_aug = (teacher_preds @ teacher_aug_preds.T)/temp
    teacher_teacher_aug_t = (teacher_aug_preds @ teacher_preds.T)/temp
    student_teacher_aug = (student_preds @ teacher_aug_preds.T)/temp
    student_teacher_aug_t = (teacher_aug_preds @ student_preds.T)/temp

    loss1 = -(student_teacher_aug.log_softmax(-1) * teacher_teacher_aug.softmax(-1)).sum(-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(-1) * teacher_teacher_aug_t.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def format_tiled_figure(images, captions, rows, cols, red_line_index=None, buffer=10, mode=0, title=None, font_size=60):
    """
    Assembles a tiled figure of images with optional captions and a red background behind a specified column or row.

    :param images: List of PIL Image objects, ordered row-wise.
    :param captions: List of captions, length and usage depends on mode.
    :param rows: Number of rows in the image grid.
    :param cols: Number of columns in the image grid.
    :param red_line_index: Index of the row or column to highlight with a red background (0-indexed).
    :param buffer: Buffer value in pixels for space between images.
    :param mode: Mode of the figure assembly.
    :param title: Title of the figure, used in mode 1 and mode 3.
    :return: PIL Image object of the assembled figure.
    """
    
    # Find the smallest width and height among all images
    min_width, min_height = min(img.size for img in images)

    # Resize all images to the smallest dimensions
    images = [img.resize((min_width, min_height), Image.ANTIALIAS) for img in images]

    # Font setup
    # font_size = 60  # Base font size for readability
    row_caption_font_size = font_size  
    title_font_size = int(1.3 * font_size) 
    title_font = ImageFont.truetype("arial.ttf", title_font_size)
    row_caption_font = ImageFont.truetype("arial.ttf", row_caption_font_size)

    # Calculate dimensions for the entire canvas
    caption_height = row_caption_font_size if mode in [0, 1] else 0
    title_height = int(title_font_size * 1.3) if mode in [1, 3] and title is not None or mode in [2] and captions is not None else 0  # Adjusted to include mode 3
    row_title_width = int(row_caption_font_size * 1.5) if mode == 3 else 0
    extra_buffer_w = buffer if (red_line_index is not None and mode in [0, 1, 2]) else 0
    extra_buffer_h = buffer if (red_line_index is not None and mode == 3) else 0

    # Calculate the total canvas width and height
    total_width = cols * (min_width + buffer) + row_title_width + buffer + extra_buffer_w
    total_height = rows * (min_height + buffer) + title_height + rows * caption_height + buffer + extra_buffer_h

    # Create a new image with a white background
    canvas = Image.new('RGB', (total_width, total_height), color='white')

    # Prepare the drawing context
    draw = ImageDraw.Draw(canvas)

    # Draw the title for modes 1 and 3
    if mode in [1, 3] and title is not None:  # Adjusted to include mode 3
        text_width, text_height = draw.textsize(title, font=title_font)
        draw.text(((total_width - text_width) // 2, (title_height - text_height) // 2), title, font=title_font, fill='black')

    # Draw red background before placing images if a red line index is specified
    if red_line_index is not None:
        if mode in [0, 1, 2]:  # Red column
            red_x = row_title_width + red_line_index * (min_width + buffer)
            red_y = title_height
            red_width = min_width + buffer * 2
            red_height = total_height - title_height
            canvas.paste(Image.new('RGB', (red_width, red_height), color='red'), (red_x, red_y))
        elif mode == 3:  # Red row
            red_x = row_title_width
            red_y = title_height + red_line_index * (min_height + buffer)
            red_width = total_width - row_title_width
            red_height = min_height + buffer * 2
            canvas.paste(Image.new('RGB', (red_width, red_height), color='red'), (red_x, red_y))

    # Insert images into the canvas
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            if idx >= len(images):
                continue

            img = images[idx]
            x = col * (min_width + buffer) + row_title_width + buffer
            y = row * (min_height + buffer) + title_height + buffer

            # Adjust the x position if there is a red column
            if mode in [0, 1, 2] and red_line_index is not None and col > red_line_index:
                x += extra_buffer_w

            # Adjust the y position if there is a red row
            if mode == 3 and red_line_index is not None and row > red_line_index:
                y += extra_buffer_h

            # Paste the image
            canvas.paste(img, (x, y))
    # Draw the vertical text for row titles if mode is 3
    if mode == 3:
        for row, caption in enumerate(captions):
            # Calculate the caption size using the default font
            width, height = row_caption_font.getsize(caption)

            text_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_image)
            draw.text((0, 0), text=caption, font=row_caption_font, fill='black')

            # Rotate the text image to be vertical
            text_image = text_image.rotate(90, expand=1)

            # Calculate the y position for the vertical text
            y = row * (min_height + buffer) + (min_width - width )//2 + title_height
            if row > 0:
                y += buffer

            # Calculate the x position, accounting for the increased text size
            x = 0

            # Paste the rotated text image onto the canvas
            canvas.paste(text_image, (x, y), text_image)

    # Draw captions for each image for modes 0 and 1
    if mode in [0, 1]:
        for idx, caption in enumerate(captions):
            col = idx % cols
            row = idx // cols
            text_width, text_height = draw.textsize(caption, font=row_caption_font)
            x = col * (min_width + buffer) + row_title_width + buffer + (min_width - text_width) // 2
            y = (row + 1) * (min_height + buffer) + title_height - text_height // 2
            draw.text((x, y), caption, font=row_caption_font, fill='black')

    # Draw column titles if mode is 2
    if mode == 2:
        for col, caption in enumerate(captions):
            text_width, text_height = draw.textsize(caption, font=row_caption_font)
            x = col * (min_width + buffer) + row_title_width + buffer + (min_width - text_width) // 2
            y = buffer
            draw.text((x, y), caption, font=row_caption_font, fill='black')

    return canvas

def condition_average(x, y, cond, nest=False):
    idx, idx_count = np.unique(cond, return_counts=True)
    idx_list = [np.array(cond)==i for i in np.sort(idx)]
    if nest:
        avg_x = torch.zeros((len(idx), idx_count.max(), x.shape[1]), dtype=torch.float32)
    else:
        avg_x = torch.zeros((len(idx), 1, x.shape[1]), dtype=torch.float32)
    arranged_y = torch.zeros((len(idx)), y.shape[1], y.shape[2], y.shape[3])
    for i, m in enumerate(idx_list):
        if nest:
            if np.sum(m) == idx_count.max():
                avg_x[i] = x[m]
            else:
                avg_x[i,:np.sum(m)] = x[m]
        else:
            avg_x[i] = torch.mean(x[m], axis=0)
        arranged_y[i] = y[m[0]]

    return avg_x, y, len(idx_count)

def condition_average_old(x, y, cond, nest=False):
    idx, idx_count = np.unique(cond, return_counts=True)
    idx_list = [np.array(cond)==i for i in np.sort(idx)]
    if nest:
        avg_x = torch.zeros((len(idx), idx_count.max(), x.shape[1]), dtype=torch.float32)
    else:
        avg_x = torch.zeros((len(idx), 1, x.shape[1]), dtype=torch.float32)
    arranged_y = torch.zeros((len(idx)), y.shape[1], y.shape[2], y.shape[3])
    for i, m in enumerate(idx_list):
        if nest:
            if np.sum(m) == idx_count.max():
                avg_x[i] = x[m]
            else:
                avg_x[i,:np.sum(m)] = x[m]
        else:
            avg_x[i] = torch.mean(x[m], axis=0)
        arranged_y[i] = y[m[0]]

    return avg_x, y, len(idx_count)

#subject: nsd subject index between 1-8
#mode: vision, imagery
#stimtype: all, simple, complex, concepts
#average: whether to average across trials, will produce x that is (stimuli, 1, voxels)
#nest: whether to nest the data according to stimuli, will produce x that is (stimuli, trials, voxels)
#data_root: path to where the dataset is saved.
def load_nsd_mental_imagery(subject, mode, stimtype="all", average=False, num_reps = 16, nest=False, snr=-1, top_n_rois=-1, samplewise=False, whole_brain=False, nsd_general=False, data_root="../dataset"):
    # This file has a bunch of information about the stimuli and cue associations that will make loading it easier
    img_stim_file = f"{data_root}/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3"
    ex_file = open(img_stim_file, 'rb')
    imagery_dict = pickle.load(ex_file)
    ex_file.close()
    # Indicates what experiments trials belong to
    exps = imagery_dict['exps']
    # Indicates the cues for different stimuli
    cues = imagery_dict['cues']
    # Maps the cues to the stimulus image information
    image_map  = imagery_dict['image_map']
    # Organize the indices of the trials according to the modality and the type of stimuli
    cond_idx = {
    'visionsimple': np.arange(len(exps))[exps=='visA'],
    'visioncomplex': np.arange(len(exps))[exps=='visB'],
    'visionconcepts': np.arange(len(exps))[exps=='visC'],
    'visionall': np.arange(len(exps))[np.logical_or(np.logical_or(exps=='visA', exps=='visB'), exps=='visC')],
    'imagerysimple': np.arange(len(exps))[np.logical_or(exps=='imgA_1', exps=='imgA_2')],
    'imagerycomplex': np.arange(len(exps))[np.logical_or(exps=='imgB_1', exps=='imgB_2')],
    'imageryconcepts': np.arange(len(exps))[np.logical_or(exps=='imgC_1', exps=='imgC_2')],
    'imageryall': np.arange(len(exps))[np.logical_or(
                                        np.logical_or(
                                            np.logical_or(exps=='imgA_1', exps=='imgA_2'),
                                            np.logical_or(exps=='imgB_1', exps=='imgB_2')),
                                        np.logical_or(exps=='imgC_1', exps=='imgC_2'))]}
    print(f"load nsd mi: nsdgeneral {nsd_general}, whole brain {whole_brain}, top n rois {top_n_rois}, samplewise {samplewise}")
    # Load normalized betas
    if whole_brain:
        x = torch.load(f"{data_root}/preprocessed_data/subject{subject}/nsd_imagery_whole_brain.pt")
    elif top_n_rois != -1:
        x = torch.load(f"{data_root}/preprocessed_data/subject{subject}/nsd_imagery_whole_brain.pt")
        x = mask_whole_brain_on_top_n_rois(subject, x, top_n_rois, samplewise, nsd_general, data_root)  
    else:
        if snr == -1.0:
            x = torch.load(f"{data_root}/preprocessed_data/subject{subject}/nsd_imagery.pt").requires_grad_(False).to("cpu")
        else:
            if not os.path.exists(f"{data_root}/preprocessed_data/subject{subject}/nsd_imagery_whole_brain.pt"):
                create_whole_region_imagery_unnormalized(subject = subject, mask=False, data_path=data_root)
                create_whole_region_imagery_normalized(subject = subject, mask=False, data_path=data_root)
            x = torch.load(f"{data_root}/preprocessed_data/subject{subject}/nsd_imagery_whole_brain.pt")
            snr_mask = calculate_snr_mask(subject, snr, data_path=data_root)
            x = x[:,snr_mask]
    # Find the trial indices conditioned on the type of trials we want to load
    cond_im_idx = {n: [image_map[c] for c in cues[idx]] for n,idx in cond_idx.items()}
    conditionals = cond_im_idx[mode+stimtype]
    # Stimuli file is of shape (18,3,425,425), these can be converted back into PIL images using transforms.ToPILImage()
    y = torch.load(f"{data_root}/nsddata_stimuli/stimuli/imagery_stimuli_18.pt").requires_grad_(False).to("cpu")
    # Prune the beta file down to specific experimental mode/stimuli type
    x = x[cond_idx[mode+stimtype]]
    # # If stimtype is not all, then prune the image data down to the specific stimuli type
    if stimtype == "simple":
        y = y[:6]
    elif stimtype == "complex":
        y = y[6:12]
    elif stimtype == "concepts":
        y = y[12:]

    # Average or nest the betas across trials
    if average or nest:
        x, y, sample_count = condition_average(x, y, conditionals, nest=nest)
    else:
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        y = y[conditionals]

    print(x.shape, y.shape)
    return x, y

#subject: nsd subject index between 1-8
#average: whether to average across trials, will produce x that is (stimuli, 1, voxels)
#nest: whether to nest the data according to stimuli, will produce x that is (stimuli, trials, voxels)
#data_root: path to where the dataset is saved.
def load_nsd_synthetic(subject, average=False, nest=False, data_root="../dataset/"):
    y = torch.zeros((284, 3, 714, 1360))
    y[:220] = torch.load(f"{data_root}/nsddata_stimuli/stimuli/nsdsynthetic/nsd_synthetic_stim_part1.pt")
    #The last 64 stimuli are slightly different for each subject, so we load these separately for each subject
    y[220:] = torch.load(f"{data_root}/nsddata_stimuli/stimuli/nsdsynthetic/nsd_synthetic_stim_part2_sub{subject}.pt")
    
    x = torch.load(f"{data_root}/preprocessed_data/subject{subject}/nsd_synthetic.pt").requires_grad_(False).to("cpu")
    conditionals = loadmat(f'{data_root}/nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat')['masterordering'][0].astype(int) - 1
    
    if average or nest:
        x, y, sample_count = condition_average(x, y, conditionals, nest=nest)
    else:
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        y = y[conditionals]
    print(x.shape, y.shape)
    return x, y    

#subject: subject index between 1-3, or the subject identifier: subj01, subj02, subj03. These are NOT the NSD subjects as this is a different datasets
#mode: vision, imagery
#mask: True or False, if true masks the betas to visual cortex, otherwise returns the whole scanned region
#stimtype: stimuli, cue, object
    # - stimuli will return the images with content that was either seen or imagined, this is what was presented to the subject in vision trials
    # - cue will return only the background images with the cue and no content, this is what was presented to the subject in imagery trials
    # - object will return only the object in the image with no cue or location brackets. This should be used for model training where we dont want the model to learn the brackets or the cue.
#average: whether to average across trials, will produce x that is (stimuli, 1, voxels)
#nest: whether to nest the data according to stimuli, will produce x that is (stimuli, trials, voxels)
    # WARNING: Not all stimuli have the same number of repeats, so the middle dimension for the trial repetitions will contain empty values for some stimuli, be sure to account for this when loading
def load_imageryrf(subject, mode, mask=True, stimtype="object", average=False, nest=False, split=False, data_root="../dataset/"):
    
    # This file has a bunch of information about the stimuli and cue associations that will make loading it easier
    img_conditional_file = f"{data_root}/imageryrf_single_trial/stimuli/imageryrf_conditions.pkl3"
    ex_file = open(img_conditional_file, 'rb')
    conditional_dict = pd.compat.pickle_compat.load(ex_file) 
    ex_file.close()
    stimuli_metadata = conditional_dict['stimuli_metadata']
    # If subject identifier is int, grab the string identifer
    if isinstance(subject, int):
        subject = f"subj0{subject}"
    subject_cond = conditional_dict[subject]
    # Indicates what experiments trials belong to
    exps = subject_cond['experiment_cond']
    # Maps the cues to the stimulus image information
    image_map  = subject_cond['stimuli_cond'].to(int)
    # Identify and condition on the stimuli that will be the test set
    test_idx = torch.tensor([0,7,15,23,35,47,51,63])
    object_idx = torch.tensor(stimuli_metadata['object_idx'].values)
    test_indices = [idx for idx, value in enumerate(object_idx) if value in test_idx]
    
    # Organize the indices of the trials according to the modality and the type of stimuli
    cond_idx = {
    'vision': np.arange(len(exps))[np.char.find(exps, 'pcp') != -1],
    'imagery': np.arange(len(exps))[np.char.find(exps, 'img') != -1],
    'all': np.arange(len(exps)),
    'visiontrain': np.arange(len(exps))[np.logical_and(np.char.find(exps, 'pcp') != -1, ~np.isin(image_map, test_indices))],
    'visiontest': np.arange(len(exps))[np.logical_and(np.char.find(exps, 'pcp') != -1, np.isin(image_map, test_indices))],
    'imagerytrain': np.arange(len(exps))[np.logical_and(np.char.find(exps, 'img') != -1, ~np.isin(image_map, test_indices))],
    'imagerytest': np.arange(len(exps))[np.logical_and(np.char.find(exps, 'img') != -1, np.isin(image_map, test_indices))],
    'alltrain': np.arange(len(exps))[~np.isin(image_map, test_indices)],
    'alltest': np.arange(len(exps))[np.isin(image_map, test_indices)]}
    # Load normalized betas
    if mask:
        x = torch.load(f"{data_root}/imageryrf_single_trial/{subject}/single_trial_betas_masked.pt").requires_grad_(False).to("cpu")
    else:
        x = torch.load(f"{data_root}/imageryrf_single_trial/{subject}/single_trial_betas.pt").requires_grad_(False).to("cpu")
    y = torch.load(f"{data_root}/imageryrf_single_trial/stimuli/{stimtype}_images.pt").requires_grad_(False).to("cpu")
    # Find the stimuli indices conditioned on the mode of trials we want to load
    if split:
        conditionals_train = image_map[cond_idx[mode+'train']]
        conditionals_test = image_map[cond_idx[mode+'test']]
        x_train = x[cond_idx[mode+'train']]
        x_test = x[cond_idx[mode+'test']]
        y_train = y[~torch.isin(torch.arange(len(y)), torch.tensor(test_indices))]
        y_test = y[test_indices]
    else:
        conditionals = image_map[cond_idx[mode]]
        # Prune the beta file down to specific experimental mode/stimuli type
        x = x[cond_idx[mode]]
        
    # Average or nest the betas across trials
    if average or nest:
        if split:
            x_train, y_train, sample_count = condition_average_old(x_train, y_train, conditionals_train, nest=nest)
            x_test, y_test, sample_count = condition_average_old(x_test, y_test, conditionals_test, nest=nest)
        else:
            x, y, sample_count = condition_average_old(x, y, conditionals, nest=nest)
    else:
        if split:
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
            y_train = y[conditionals_train]
            y_test = y[conditionals_test]
            
        else:
            x = x.reshape((x.shape[0], x.shape[1]))
            y = y[conditionals]
    
    if split:
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        return x_train, y_train, x_test, y_test
    else:
        print(x.shape, y.shape)
        return x, y
    
    
def read_betas(subject, session_index, trial_index=[], data_type='betas_fithrf_GLMdenoise_RR', data_format='fsaverage', mask=None, data_path="../dataset"):
        """read_betas read betas from MRI files

        Parameters
        ----------
        subject : str
            subject identifier, such as 'subj01'
        session_index : int
            which session, counting from 1
        trial_index : list, optional
            which trials from this session's file to return, by default [], which returns all trials
        data_type : str, optional
            which type of beta values to return from ['betas_assumehrf', 'betas_fithrf', 'betas_fithrf_GLMdenoise_RR', 'restingbetas_fithrf'], by default 'betas_fithrf_GLMdenoise_RR'
        data_format : str, optional
            what type of data format, from ['fsaverage', 'func1pt8mm', 'func1mm'], by default 'fsaverage'
        mask : numpy.ndarray, if defined, selects 'mat' data_format, needs volumetric data_format
            binary/boolean mask into mat file beta data format.

        Returns
        -------
        numpy.ndarray, 2D (fsaverage) or 4D (other data formats)
            the requested per-trial beta values
        """

        data_folder = f'{data_path}/nsddata_betas/ppdata/{subject}/{data_format}/{data_type}'
        
        si_str = str(session_index).zfill(2)

        out_data = nb.load(
            op.join(data_folder, f'betas_session{si_str}.nii.gz')).get_fdata()

        if len(trial_index) == 0:
            trial_index = slice(0, out_data.shape[-1])

        return out_data[..., trial_index]


def create_whole_region_unnormalized(subject: int = 1, include_heldout: bool = True, 
                                     mask_nsd_general: bool = False, data_path="../dataset") -> None:
    """Creates and saves an unnormalized whole region tensor for a given subject.

    This function loads, processes, and saves whole region neural data for a given subject. 
    The data can be optionally masked using the NSD general mask, and include held-out sessions.

    Args:
        subject (int, optional): The subject number (1-8). Defaults to 1.
        include_heldout (bool, optional): Whether to include held-out data. Defaults to True.
        mask_nsd_general (bool, optional): Whether to apply the NSD general mask. Defaults to False.
        data_path (str, optional): The path to the data directory. Defaults to "../dataset".

    Returns:
        None: The function saves the processed tensor to a file and does not return anything.
    """
    
    os.makedirs(f"{data_path}/preprocessed_data/subject{subject}/", exist_ok=True)

    # Determine the output file path and the number of scans based on function parameters.
    if include_heldout and mask_nsd_general:
        file_path = f"{data_path}/preprocessed_data/subject{subject}/nsd_general_unnormalized_include_heldout.pt"
        num_scans = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}
    elif include_heldout and not mask_nsd_general:
        file_path = f"{data_path}/preprocessed_data/subject{subject}/whole_brain_unnormalized_include_heldout.pt"
        num_scans = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}
    elif not include_heldout and not mask_nsd_general:
        file_path = f"{data_path}/preprocessed_data/subject{subject}/whole_brain_unnormalized.pt"
        num_scans = {1: 40, 2: 40, 3: 32, 4: 30, 5: 40, 6: 32, 7: 40, 8: 30}
    else:
        file_path = f"{data_path}/preprocessed_data/subject{subject}/nsd_general_unnormalized.pt"
        num_scans = {1: 37, 2: 37, 3: 32, 4: 30, 5: 37, 6: 32, 7: 37, 8: 30}
    
    # If the file already exists, exit the function
    if os.path.exists(file_path):
        return

    # Apply the NSD general mask if required.
    if mask_nsd_general:
        nsd_general = nb.load(f"{data_path}/nsddata/ppdata/subj0{subject}/func1pt8mm/roi/nsdgeneral.nii.gz").get_fdata()
        nsd_general = np.nan_to_num(nsd_general)
        mask = nsd_general == 1.0
    else:
        brainmask_inflated = nb.load(f"{data_path}/nsddata/ppdata/subj0{subject}/func1pt8mm/roi/brainmask_inflated_1.0.nii").get_fdata()
        brainmask_inflated = np.nan_to_num(brainmask_inflated)
        mask = brainmask_inflated == 1.0
        
    layer_size = np.sum(mask == True)
    
    data = num_scans[subject]
    whole_region = torch.zeros((750 * data, layer_size))

    mask = np.nan_to_num(mask)
    mask = np.array(mask.flatten(), dtype=bool)
    
    # Loads the full collection of beta sessions for subject 1
    for i in tqdm(range(1, data + 1), desc="Loading raw scanning session data"):
        beta = read_betas(subject="subj0" + str(subject), 
                                session_index=i, 
                                trial_index=[], # Empty list as index means get all 750 scans for this session (trial --> scan)
                                data_type="betas_fithrf_GLMdenoise_RR",
                                data_format='func1pt8mm',
                                data_path=data_path)
            
        # Reshape the beta trails to be flattened. 
        beta = beta.reshape((mask.shape[0], beta.shape[3]))

        for j in range(beta.shape[1]):

            # Grab the current beta trail. 
            current_scan = beta[:, j]
            
            # One scan session. 
            single_scan = torch.from_numpy(current_scan)

            # Discard the unmasked values and keeps the masked values. 
            whole_region[j + (i-1)*beta.shape[1]] = single_scan[mask]
            
    # Save the tensor into the data directory. 
    torch.nan_to_num(whole_region)
    torch.save(whole_region, file_path)

def zscore(x, mean=None, stddev=None, return_stats=False):
    if mean is not None:
        m = mean
    else:
        m = torch.mean(x, axis=0, keepdims=True)
    if stddev is not None:
        s = stddev
    else:
        s = torch.std(x, axis=0, keepdims=True)
    if return_stats:
        return (x - m)/(s+1e-6), m, s
    else:
        return (x - m)/(s+1e-6)
    
def create_whole_region_normalized(subject = 1, include_heldout=False, mask_nsd_general=False, data_path="../dataset/"):
        
    if include_heldout and mask_nsd_general:
        file = f"{data_path}/preprocessed_data/subject{subject}/nsd_general_include_heldout.pt"
        
        # File has already been created
        if os.path.exists(file): return
        
        whole_region = torch.load(f"{data_path}/preprocessed_data/subject{subject}/nsd_general_unnormalized_include_heldout.pt")
        numScans = {1: 40, 2: 40, 3:32, 4: 30, 5:40, 6:32, 7:40, 8:30}
        
    elif include_heldout and not mask_nsd_general:
        file = f"{data_path}/preprocessed_data/subject{subject}/whole_brain_include_heldout.pt"
        
        # File has already been created
        if os.path.exists(file): return
        
        whole_region = torch.load(f"{data_path}/preprocessed_data/subject{subject}/whole_brain_unnormalized_include_heldout.pt")
        numScans = {1: 40, 2: 40, 3:32, 4: 30, 5:40, 6:32, 7:40, 8:30}
        
    elif not include_heldout and not mask_nsd_general:
        file = f"{data_path}/preprocessed_data/subject{subject}/whole_brain.pt"
        
        # File has already been created
        if os.path.exists(file): return
        
        whole_region = torch.load(f"{data_path}/preprocessed_data/subject{subject}/whole_brain_unnormalized.pt")
        numScans = {1: 40, 2: 40, 3:32, 4: 30, 5:40, 6:32, 7:40, 8:30}
        
    else:
        file = f"{data_path}/preprocessed_data/subject{subject}/nsd_general.pt"
        
        # File has already been created
        if os.path.exists(file): return
        
        whole_region = torch.load(f"{data_path}/preprocessed_data/subject{subjec}/nsd_general_unnormalized.pt")
        numScans = {1: 37, 2: 37, 3:32, 4: 30, 5:37, 6:32, 7:37, 8:30}
    
    whole_region_norm = torch.zeros_like(whole_region)
    
    stim_descriptions = pd.read_csv(f'{data_path}/nsddata/experiments/nsd/nsd_stim_info_merged.csv', index_col=0)
    subj_train = stim_descriptions[(stim_descriptions[f'subject{subject}'] != 0) & (stim_descriptions['shared1000'] == False)]
    train_ids = []
    
    for i in range(subj_train.shape[0]):
        for j in range(3):
            scanID = subj_train.iloc[i][f'subject{subject}_rep{j}'] - 1
            if scanID < numScans[subject]*750:
                train_ids.append(scanID)
    normalizing_data = whole_region[torch.tensor(train_ids)]
    print(normalizing_data.shape, whole_region.shape)
    
    # Normalize the data using Z scoring method for each voxel
    for i in range(normalizing_data.shape[1]):
        voxel_mean, voxel_std = torch.mean(normalizing_data[:, i]), torch.std(normalizing_data[:, i])  
        normalized_voxel = (whole_region[:, i] - voxel_mean) / voxel_std
        whole_region_norm[:, i] = normalized_voxel

    # Save the tensor of normalized data
    torch.save(whole_region_norm, file)
    convert_from_pt_to_hdf5(file, f"{data_path}/betas_all_whole_brain_subj{subject:02d}_fp32_renorm.hdf5")
    
def create_whole_region_imagery_unnormalized(subject = 1, mask=True, GLMdenoise=True, data_path="../dataset/"):
    
    os.makedirs(f"{data_path}/preprocessed_data/subject{subject}/", exist_ok=True)
    if GLMdenoise:
        beta_file = f"{data_path}/nsddata_betas/ppdata/subj0{subject}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz"
    else:
        file += "_b2"
        beta_file = f"{data_path}/nsddata_betas/ppdata/subj0{subject}/func1pt8mm/nsdimagerybetas_fithrf/betas_nsdimagery.nii.gz"

    imagery_betas = nb.load(beta_file).get_fdata()

    imagery_betas = imagery_betas.transpose((3,0,1,2))
    if mask:
        file = f"{data_path}/preprocessed_data/subject{subject}/nsd_imagery_unnormalized.pt"
        nsd_general = nb.load(f"{data_path}/nsddata/ppdata/subj0{subject}/func1pt8mm/roi/nsdgeneral.nii.gz").get_fdata()
        nsd_general = np.where(nsd_general==1.0, True, False)
        nsd_general_mask = np.nan_to_num(nsd_general)
        nsd_mask = np.array(nsd_general_mask.flatten(), dtype=bool)
        whole_region = torch.from_numpy(imagery_betas.reshape((len(imagery_betas), -1))[:,nsd_general.flatten()].astype(np.float32))
    else:
        file = f"{data_path}/preprocessed_data/subject{subject}/nsd_imagery_unnormalized_whole_brain.pt"
        whole_brain = nb.load(f"{data_path}/nsddata/ppdata/subj0{subject}/func1pt8mm/roi/brainmask_inflated_1.0.nii").get_fdata()
        whole_brain = np.where(whole_brain==1.0, True, False)
        whole_brain_mask = np.nan_to_num(whole_brain)
        whole_brain_mask = np.array(whole_brain_mask.flatten(), dtype=bool)
        whole_region = torch.from_numpy(imagery_betas.reshape((len(imagery_betas), -1))[:,whole_brain_mask.flatten()].astype(np.float32))
    
    torch.save(whole_region, file)
    return whole_region

def convert_from_pt_to_hdf5(load_data_path="../dataset/", save_data_path="../dataset/"):
    
    # Load the tensor
    tensor = torch.load(load_data_path).requires_grad_(False).to("cpu")
    
    # Convert the tensor to a numpy array (h5py works with numpy arrays)
    tensor_numpy = tensor.numpy()
    
    # Save the tensor to the specified HDF5 format
    with h5py.File(save_data_path, 'w') as hdf:
        hdf.create_dataset('betas', data=tensor_numpy)
    
    
def create_whole_region_imagery_normalized(subject = 1, mask=True, GLMdenoise=True, data_path="../dataset/"):
    img_stim_file = f"{data_path}/nsddata_stimuli/stimuli/nsd/nsdimagery_stimuli.pkl3"
    ex_file = open(img_stim_file, 'rb')
    imagery_dict = pickle.load(ex_file)
    ex_file.close()
    exps = imagery_dict['exps']
    cues = imagery_dict['cues']
    meta_cond_idx = {
        'visA': np.arange(len(exps))[exps=='visA'],
        'visB': np.arange(len(exps))[exps=='visB'],
        'visC': np.arange(len(exps))[exps=='visC'],
        'imgA_1': np.arange(len(exps))[exps=='imgA_1'],
        'imgA_2': np.arange(len(exps))[exps=='imgA_2'],
        'imgB_1': np.arange(len(exps))[exps=='imgB_1'],
        'imgB_2': np.arange(len(exps))[exps=='imgB_2'],
        'imgC_1': np.arange(len(exps))[exps=='imgC_1'],
        'imgC_2': np.arange(len(exps))[exps=='imgC_2'],
        'attA': np.arange(len(exps))[exps=='attA'],
        'attB': np.arange(len(exps))[exps=='attB'],
        'attC': np.arange(len(exps))[exps=='attC'],
    }
    unnormalized_file = f"{data_path}/preprocessed_data/subject{subject}/nsd_imagery_unnormalized"
    output_file = f"{data_path}/preprocessed_data/subject{subject}/nsd_imagery"
    if not GLMdenoise:
        unnormalized_file += "_b2"
        output_file += "_b2"
    if not mask:
        unnormalized_file += "_whole_brain"
        output_file += "_whole_brain"
    whole_region = torch.load(unnormalized_file + ".pt")
    whole_region = whole_region / 300.
    whole_region_norm = torch.zeros_like(whole_region)
            
    # Normalize the data using Z scoring method for each voxel
    for c,idx in meta_cond_idx.items():
        whole_region_norm[idx] = zscore(whole_region[idx])

    # Save the tensor of normalized data
    torch.save(whole_region_norm, output_file + ".pt")
    # Delete NSD unnormalized file after the normalized data is created. 
    if(os.path.exists(unnormalized_file + ".pt")):
        os.remove(unnormalized_file + ".pt")
    
def calculate_snr(betas):
    averaged_betas = torch.mean(betas, dim=1)
    signal = torch.var(averaged_betas, dim=0)
    trial_variance = torch.var(betas, dim=1)
    noise = torch.mean(trial_variance, dim=0)
    snr = signal / noise
    snr = torch.nan_to_num(snr)
    return snr, signal, noise

def create_snr_betas(subject=1, data_type=torch.float16, data_path="../dataset/", threshold=-1.0):
    
    if threshold != -1.0:
        create_whole_region_unnormalized(subject = subject, include_heldout=True, mask_nsd_general=False, data_path=data_path)
        create_whole_region_normalized(subject = subject, include_heldout=True, mask_nsd_general=False, data_path=data_path)
        # Load the tensor from the HDF5 file
        with h5py.File(f'{data_path}/betas_all_whole_brain_subj{subject:02d}_fp32_renorm.hdf5', 'r') as f:
            betas = f['betas'][:]
            betas = torch.from_numpy(betas).to("cpu")
    
        snr_mask = calculate_snr_mask(subject, threshold, betas=betas, data_path=data_path)
        
        # Filter out the zero columns
        betas = betas[:, snr_mask]
        
    else:      
        with h5py.File(f'{data_path}/betas_all_subj{subject:02d}_fp32_renorm.hdf5', 'r') as f:
            betas = f['betas'][:]
            betas = torch.from_numpy(betas).to("cpu")
        
    return betas.to(data_type)

def load_nsd(subject, betas=None, data_path="../dataset/"):
    # Load betas if not provided
    if betas is None:
        with h5py.File(f'{data_path}/betas_all_subj{subject:02d}_fp32_renorm.hdf5', 'r') as f:
            betas = f['betas'][:]
            betas = torch.from_numpy(betas).to("cpu")

    # Load stimulus descriptions
    stim_descriptions = pd.read_csv(
        f"{data_path}/nsddata/experiments/nsd/nsd_stim_info_merged.csv", index_col=0
    )

    # Define repeat columns
    rep_columns = [f"subject{subject}_rep{j}" for j in range(3)]

    # Filter training data (exclude shared1000 trials)
    subj_train = stim_descriptions[
        (stim_descriptions[f"subject{subject}"] != 0) & (stim_descriptions["shared1000"] == False)
    ]

    # Get the scan IDs for the three repeats in training data
    scan_ids_train = subj_train[rep_columns].values - 1  # Convert to zero-based indices

    # Flatten the scan IDs for training data
    flat_scan_ids_train = scan_ids_train.flatten()

    # Create an array of nsd IDs repeated for each repeat in training data
    nsd_ids_train = subj_train["nsdId"].values
    repeated_nsd_ids_train = np.repeat(nsd_ids_train, 3)

    # Handle missing values and invalid indices in training data
    valid_mask_train = (
        (~np.isnan(flat_scan_ids_train))
        & (flat_scan_ids_train >= 0)
        & (flat_scan_ids_train < betas.shape[0])
    )
    valid_scan_ids_train = flat_scan_ids_train[valid_mask_train].astype(int)
    valid_nsd_ids_train = repeated_nsd_ids_train[valid_mask_train].astype(int)

    # Extract the corresponding brain activity data for training data
    x_train = betas[valid_scan_ids_train]

    # Filter test data (include shared1000 trials)
    subj_test = stim_descriptions[
        (stim_descriptions[f"subject{subject}"] != 0) & (stim_descriptions["shared1000"] == True)
    ]

    # Get the scan IDs for the three repeats in test data
    scan_ids_test = subj_test[rep_columns].values - 1  # Convert to zero-based indices

    # Handle missing values and invalid indices in test data
    valid_mask_test = (
        (~np.isnan(scan_ids_test))
        & (scan_ids_test >= 0)
        & (scan_ids_test < betas.shape[0])
    )
    scan_ids_test[~valid_mask_test] = -1  # Mark invalid indices with -1

    # Prepare to extract betas for test data
    num_test_trials, num_repeats = scan_ids_test.shape
    betas_test = torch.zeros((num_test_trials, num_repeats, betas.shape[1]), dtype=betas.dtype)

    # Extract betas for valid scan IDs
    for i in range(num_test_trials):
        for j in range(num_repeats):
            scan_id = scan_ids_test[i, j]
            if scan_id >= 0:
                betas_test[i, j] = betas[int(scan_id)]

    # Create a mask tensor for valid betas
    valid_mask_test_tensor = torch.from_numpy(valid_mask_test.astype(np.float32))

    # Sum over repeats
    betas_test_sum = betas_test.sum(dim=1)  # Shape: (1000, voxels)

    # Count valid repeats for each trial
    valid_counts = valid_mask_test.sum(axis=1)  # Shape: (1000,)
    valid_counts_tensor = torch.from_numpy(valid_counts).float().unsqueeze(1)

    # Avoid division by zero
    valid_counts_tensor[valid_counts_tensor == 0] = 1

    # Compute the average over valid repeats
    x_test = betas_test_sum / valid_counts_tensor

    # Set x_test to zero where there are no valid repeats
    zero_counts = (valid_counts == 0)
    if zero_counts.any():
        x_test[zero_counts] = 0

    # Get nsd IDs for test data
    test_nsd_ids = subj_test["nsdId"].values.astype(int)

    return x_train, valid_nsd_ids_train, x_test, test_nsd_ids

def load_subject_masks(subject_ids, data_path, nsd_general=False):
    subject_masks = {}

    # Preload masks for all subjects
    for subject_id in subject_ids:
        if nsd_general:
            mask_path = f"{data_path}/combined_masks/{subject_id}_combined_mask_nsd_general.nii.gz"
        else:
            mask_path = f"{data_path}/combined_masks/{subject_id}_combined_mask.nii.gz"
        mask = nb.load(mask_path).get_fdata()

        brainmask_path = f"{data_path}/nsddata/ppdata/{subject_id}/func1pt8mm/roi/brainmask_inflated_1.0.nii"
        brainmask_inflated = nb.load(brainmask_path).get_fdata()

        # Clean up brainmask
        brainmask_inflated = np.nan_to_num(brainmask_inflated)
        brainmask_inflated = np.where(brainmask_inflated == 1.0, True, False)

        mask = mask[brainmask_inflated]

        # Load ROI labels
        if nsd_general:
            labels_path = f"{data_path}/combined_masks/{subject_id}_labels_nsd_general.txt"
        else:
            labels_path = f"{data_path}/combined_masks/{subject_id}_labels.txt"
        label_to_roi = {}
        with open(labels_path, 'r') as label_file:
            for line in label_file:
                idx, roi_name = line.strip().split(": ")
                if roi_name != 'No Mask':
                    label_to_roi[int(idx)] = roi_name

        # Create filtered masks for ROIs
        filtered_mask = {roi: mask == number for number, roi in label_to_roi.items()}
        subject_masks[subject_id] = {"filtered_mask": filtered_mask, "label_to_roi": label_to_roi}

    return subject_masks


def mask_whole_brain_on_top_n_rois(excluded_subject, betas, top_n_rois, samplewise, nsd_general, data_path): 
    print(f"mask_whole_brain_on_top_n_rois: nsdgeneral {nsd_general}, top n rois {top_n_rois}, samplewise {samplewise}")
    subject_ids = [f'subj0{i}' for i in range(1, 9)]
    subject_masks = load_subject_masks(subject_ids, data_path, nsd_general)
    excluded_subject_mask = subject_masks[f'subj0{excluded_subject}']['filtered_mask']
    
    # print(len(excluded_subject_mask.keys()), excluded_subject_mask.keys())
    rank_order_rois = {}
    
    # Load rank order ROIs from JSON file
    # print(nsd_general, samplewise)
    if nsd_general:
        with open(f'{data_path}/subj0{excluded_subject}_sorted_rois_rank_order_samplewise_nsd_general.json', 'r') as file:
            rank_order_rois = json.load(file)
    elif samplewise:
        with open(f'{data_path}/subj0{excluded_subject}_sorted_rois_rank_order_samplewise.json', 'r') as file:
            rank_order_rois = json.load(file)
    else:
        with open(f'{data_path}/subj0{excluded_subject}_sorted_rois_rank_order_voxelwise.json', 'r') as file:
            rank_order_rois = json.load(file)
    
    rank_order_rois_keys = list(rank_order_rois.keys())
    # Create initial ROI mask
    roi_mask = np.logical_or(excluded_subject_mask[rank_order_rois_keys[0]], excluded_subject_mask[rank_order_rois_keys[0]])
    
    # Apply ROI masks for top_n_rois
    for i in range(top_n_rois):
        roi_mask = np.logical_or(roi_mask, excluded_subject_mask[rank_order_rois_keys[i]])
    
    # Convert to PyTorch tensor
    roi_mask = torch.from_numpy(roi_mask).to("cpu")
    
    # Apply mask to betas
    betas = betas[..., roi_mask]
    
    return betas


def load_subject_based_on_rank_order_rois(excluded_subject=1, data_type=torch.float16, top_n_rois=-1, samplewise=False, nsd_general=False, data_path="../dataset/"):
    
    if top_n_rois != -1.0:
        
        # Load the betas for the excluded subject
        with h5py.File(f'{data_path}/betas_all_whole_brain_subj{excluded_subject:02d}_fp32_renorm.hdf5', 'r') as f:
            betas = f['betas'][:]
            betas = torch.from_numpy(betas).to("cpu")
    
        betas = mask_whole_brain_on_top_n_rois(excluded_subject, betas, top_n_rois, samplewise, nsd_general=nsd_general, data_path=data_path)
    
    else:              
        # Load betas without applying any ROI masking
        with h5py.File(f'{data_path}/betas_all_subj{excluded_subject:02d}_fp32_renorm.hdf5', 'r') as f:
            betas = f['betas'][:]
            betas = torch.from_numpy(betas).to("cpu")
    
    return betas.to(data_type)

def load_nsd(subject, betas=None, data_path="../dataset/"):
    # Load betas if not provided
    if betas is None:
        with h5py.File(f'{data_path}/betas_all_subj{subject:02d}_fp32_renorm.hdf5', 'r') as f:
            betas = f['betas'][:]
            betas = torch.from_numpy(betas).to("cpu").to(torch.float16)

    # Load stimulus descriptions
    stim_descriptions = pd.read_csv(
        f"{data_path}/nsddata/experiments/nsd/nsd_stim_info_merged.csv", index_col=0
    )

    # Define repeat columns
    rep_columns = [f"subject{subject}_rep{j}" for j in range(3)]

    # Filter training data (exclude shared1000 trials)
    subj_train = stim_descriptions[
        (stim_descriptions[f"subject{subject}"] != 0) & (stim_descriptions["shared1000"] == False)
    ]

    # Get the scan IDs for the three repeats in training data
    scan_ids_train = subj_train[rep_columns].values - 1  # Convert to zero-based indices

    # Flatten the scan IDs for training data
    flat_scan_ids_train = scan_ids_train.flatten()

    # Create an array of nsd IDs repeated for each repeat in training data
    nsd_ids_train = subj_train["nsdId"].values
    repeated_nsd_ids_train = np.repeat(nsd_ids_train, 3)

    # Handle missing values and invalid indices in training data
    valid_mask_train = (
        (~np.isnan(flat_scan_ids_train))
        & (flat_scan_ids_train >= 0)
        & (flat_scan_ids_train < betas.shape[0])
    )
    valid_scan_ids_train = flat_scan_ids_train[valid_mask_train].astype(int)
    valid_nsd_ids_train = repeated_nsd_ids_train[valid_mask_train].astype(int)

    # Extract the corresponding brain activity data for training data
    x_train = betas[valid_scan_ids_train]

    # Filter test data (include shared1000 trials)
    subj_test = stim_descriptions[
        (stim_descriptions[f"subject{subject}"] != 0) & (stim_descriptions["shared1000"] == True)
    ]

    # Get the scan IDs for the three repeats in test data
    scan_ids_test = subj_test[rep_columns].values - 1  # Convert to zero-based indices

    # Handle missing values and invalid indices in test data
    valid_mask_test = (
        (~np.isnan(scan_ids_test))
        & (scan_ids_test >= 0)
        & (scan_ids_test < betas.shape[0])
    )
    scan_ids_test[~valid_mask_test] = -1  # Mark invalid indices with -1

    # Prepare to extract betas for test data
    num_test_trials, num_repeats = scan_ids_test.shape
    betas_test = torch.zeros((num_test_trials, num_repeats, betas.shape[1]), dtype=betas.dtype)

    # Extract betas for valid scan IDs
    for i in range(num_test_trials):
        for j in range(num_repeats):
            scan_id = scan_ids_test[i, j]
            if scan_id >= 0:
                betas_test[i, j] = betas[int(scan_id)]

    # Create a mask tensor for valid betas
    valid_mask_test_tensor = torch.from_numpy(valid_mask_test.astype(np.float32))

    # Sum over repeats
    betas_test_sum = betas_test.sum(dim=1)  # Shape: (1000, voxels)

    # Count valid repeats for each trial
    valid_counts = valid_mask_test.sum(axis=1)  # Shape: (1000,)
    valid_counts_tensor = torch.from_numpy(valid_counts).float().unsqueeze(1)

    # Avoid division by zero
    valid_counts_tensor[valid_counts_tensor == 0] = 1

    # Compute the average over valid repeats
    x_test = betas_test_sum / valid_counts_tensor

    # Set x_test to zero where there are no valid repeats
    zero_counts = (valid_counts == 0)
    if zero_counts.any():
        x_test[zero_counts] = 0

    # Get nsd IDs for test data
    test_nsd_ids = subj_test["nsdId"].values.astype(int)

    return x_train, valid_nsd_ids_train, x_test, test_nsd_ids


def calculate_snr_mask(subject, threshold, betas=None, data_path="../dataset/"):
    
    if betas is None:
        beta_file = f"{data_path}/preprocessed_data/subject{subject}/whole_brain_include_heldout.pt"
        x = torch.load(beta_file).requires_grad_(False).to("cpu")
    else:
        x = betas
        
    # Load stimulus descriptions
    stim_descriptions = pd.read_csv(f"{data_path}/nsddata/experiments/nsd/nsd_stim_info_merged.csv", index_col=0)

    # Filter training and testing data
    subj_train = stim_descriptions[
        (stim_descriptions[f'subject{subject}'] != 0) & (stim_descriptions['shared1000'] == False)
    ]
    subj_test = stim_descriptions[
        (stim_descriptions[f'subject{subject}'] != 0) & (stim_descriptions['shared1000'] == True)
    ]

    # Prepare the scan IDs
    rep_columns = [f'subject{subject}_rep{j}' for j in range(3)]
    scanIds = subj_train[rep_columns].values - 1  # Convert to zero-based indices

    # Handle missing values and invalid indices
    scanIds = np.where(np.isnan(scanIds), -1, scanIds).astype(int)
    valid_mask = (scanIds >= 0) & (scanIds < x.shape[0])

    # Flatten arrays for advanced indexing
    flat_scanIds = scanIds.flatten()
    flat_valid_mask = valid_mask.flatten()

    # Indices of valid scan IDs
    valid_indices = np.where(flat_valid_mask)[0]
    valid_scanIds = flat_scanIds[valid_indices]

    # Map valid_indices back to (i, j) indices
    i_indices = valid_indices // 3
    j_indices = valid_indices % 3

    # Retrieve corresponding x values
    x_values = x[valid_scanIds]

    # Initialize x_train tensor
    x_train = torch.zeros((subj_train.shape[0], 3, x.shape[1]), dtype=x.dtype)

    # Assign x_values to x_train at the correct positions
    x_train[i_indices, j_indices, :] = x_values
                            
    snr, signal, noise = calculate_snr(x_train)
    condition = snr > threshold
    snr_tensor = torch.where(condition, x, torch.tensor(0.0))
    snr_mask = (snr_tensor != 0.0).any(dim=0)

    return snr_mask


def get_kastner_masks(subject, data_path):
    kastner_labels = f"{data_path}/nsddata/freesurfer/subj0{subject}/label/Kastner2015.mgz.ctab"
    brainmask_inflated = nib.load(f"{data_path}/nsddata/ppdata/subj0{subject}/func1pt8mm/roi/brainmask_inflated_1.0.nii").get_fdata()
    brainmask_inflated = np.nan_to_num(brainmask_inflated)
    brainmask_inflated = np.where(brainmask_inflated==1.0, True, False)
    
    masks = []
    for hemi in ["lh", "rh"]:
        masks.append(nib.load(f"{data_path}/nsddata/ppdata/subj0{subject}/func1pt8mm/roi/{hemi}.Kastner2015.nii.gz").get_fdata())
    kastner_mask = masks[0] + masks[1]
    kastner_mask = kastner_mask[brainmask_inflated]
    with open(kastner_labels, 'r') as file:
        labels = file.read().splitlines()
    kastner_mask_labeled = {}
    for label in labels[1:]:
        label = label.split(" ")
        kastner_mask_labeled[label[1].strip()] = np.where(kastner_mask==int(label[0]), True, False)
        
    return kastner_mask_labeled
