import os
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import PIL
from PIL import Image
import clip
from functools import partial
import random
import json
from typing import Union, List, Optional
from tqdm import tqdm
import utils
import open_clip
import diffusers
from diffusers.models.vae import Decoder

class BrainNetwork(nn.Module):
    def __init__(self, h=4096, in_dim=15724, out_dim=768, seq_len=2, n_blocks=4, drop=.15, clip_size=768, blurry_recon=True, text_clip=False, clip_scale=1):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.blurry_recon = blurry_recon
        self.text_clip = text_clip
        self.clip_scale = clip_scale
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)
        if text_clip:
            self.backbone_linear_txt = nn.Linear(h * seq_len, 77*clip_size, bias=True) 
            self.clip_txt_proj = self.projector(clip_size, clip_size, h=clip_size)
        if self.blurry_recon:
            self.blin1 = nn.Linear(h*seq_len,4*28*28,bias=True)
            self.bdropout = nn.Dropout(.3)
            self.bnorm = nn.GroupNorm(1, 64)
            self.bupsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[32, 64, 128],
                layers_per_block=1,
            )
            self.b_maps_projector = nn.Sequential(
                nn.Conv2d(64, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=True),
            )
            
    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
    
    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )
        
    def forward(self, x):
        # make empty tensors
        c,c_txt,b = torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([[0.],[0.]])
        
        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0,2,1)
        for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0,2,1)
            
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0,2,1)
            
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        if self.text_clip:
            backbone_txt = self.backbone_linear_txt(x).reshape(len(x), -1, self.clip_size)
        if self.clip_scale>0:
            c = self.clip_proj(backbone)
            if self.text_clip:
                c_txt = self.clip_txt_proj(backbone_txt)

        if self.blurry_recon:
            b = self.blin1(x)
            b = self.bdropout(b)
            b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
            b = self.bnorm(b)
            b_aux = self.b_maps_projector(b).flatten(2).permute(0,2,1)
            b_aux = b_aux.view(len(b_aux), 49, 512)
            b = (self.bupsampler(b), b_aux)
        
        return backbone, c_txt, c, b

    
    
class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False,
                 hidden_state=False, device=torch.device('cpu')):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64"
        print(clip_variant, device)
        
        if clip_variant=="ViT-L/14" and hidden_state:
            from transformers import CLIPVisionModelWithProjection, CLIPModel,CLIPTokenizer,CLIPProcessor
            image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
            # text_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
            # image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",cache_dir="/fsx/proj-medarc/fmri/cache")
            #from transformers import CLIPVisionModelWithProjection
            #sd_cache_dir = '/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
            #image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_cache_dir, subfolder='image_encoder').eval()
            image_encoder = image_encoder.to(device)
            # tokenizer =tokenizer.to(device)
            # processor = processor.to(device)
            model = model.to(device)
            for param in image_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            for param in model.parameters():
                param.requires_grad = False # dont need to calculate gradients
            # for param in text_encoder.parameters():
            #     param.requires_grad = False # dont need to calculate gradients
            self.tokenizer = tokenizer
            self.processor = processor
            self.model = model
            self.image_encoder = image_encoder
        elif hidden_state:
            raise Exception("hidden_state embeddings only works with ViT-L/14 right now")
        
        clip_model, preprocess = clip.load(clip_variant, device=device)
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        self.clip = clip_model
        self.clip_variant = clip_variant
        if clip_variant == "RN50x64":
            self.clip_size = (448,448)
        else:
            self.clip_size = (224,224)
            
        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc
        self.hidden_state = hidden_state
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.device= device
        
        def versatile_normalize_embeddings(encoder_output):
            # original code:
            # ---------------------------------------------------------------
            # outputs = self.model.vision_model(pixel_values=pixels)
            # z = outputs.last_hidden_state
            # z = self.model.vision_model.post_layernorm(z)
            # z = self.model.visual_projection(z)
            # z_pooled = z[:, 0:1]
            # z = z / torch.norm(z_pooled, dim=-1, keepdim=True)
            # ---------------------------------------------------------------
            embeds = encoder_output.last_hidden_state
            embeds = image_encoder.vision_model.post_layernorm(embeds)
            embeds = image_encoder.visual_projection(embeds)
            return embeds
        self.versatile_normalize_embeddings = versatile_normalize_embeddings

        def versatile_normalize_text_embeddings(encoder_output):
            # original code:
            # ---------------------------------------------------------------
            # outputs = self.model.text_model(input_ids=tokens)
            # z = self.model.text_projection(outputs.last_hidden_state)
            # z_pooled = self.model.text_projection(outputs.pooler_output)
            # z = z / torch.norm(z_pooled.unsqueeze(1), dim=-1, keepdim=True)
            # ---------------------------------------------------------------
            txt_embeds = self.model.text_projection(encoder_output.last_hidden_state)
            txt_pooled = self.model.text_projection(encoder_output.pooler_output)
            return txt_embeds, txt_pooled
        self.versatile_normalize_text_embeddings = versatile_normalize_text_embeddings


    def resize_image(self, image):
        # note: antialias should be False if planning to use Pinkney's Image Variation SD model
        return transforms.Resize(self.clip_size)(image.to(self.device))

    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        if self.hidden_state:
            # clip_emb = self.preprocess((image/1.5+.25).to(self.device)) # for some reason the /1.5+.25 prevents oversaturation
            clip_emb = self.preprocess((image).to(self.device))
            clip_emb = self.image_encoder(clip_emb)
            clip_emb = self.versatile_normalize_embeddings(clip_emb)
        else:
            clip_emb = self.preprocess(image.to(self.device))
            clip_emb = self.clip.encode_image(clip_emb)
        # input is now in CLIP space, but mind-reader preprint further processes embeddings:
        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5)
        if self.norm_embs:
            if self.hidden_state:        
                # normalize all tokens by cls token's norm
                clip_emb = clip_emb / torch.norm(clip_emb[:, 0], dim=-1).reshape(-1, 1, 1)
            else:
                clip_emb = nn.functional.normalize(clip_emb, dim=-1)
        return clip_emb

    def embed_text(self, text_samples):
        if self.hidden_state:
            batch_encoding = self.tokenizer(text_samples, truncation=True, max_length=77, return_length=True,
            return_overflowing_tokens=False, padding="max_length", return_tensors="pt").to(self.device)
            tokens = batch_encoding["input_ids"].to(self.device)
            clip_emb = self.model.text_model(input_ids=tokens)
            clip_emb, clip_pooled = self.versatile_normalize_text_embeddings(clip_emb)  
        else:          
            clip_text = clip.tokenize(text_samples).to(self.device)
            clip_text = self.clip.encode_text(clip_text)
        if self.clamp_embs:
            clip_text = torch.clamp(clip_text, -1.5, 1.5)
        if self.norm_embs:
            if self.hidden_state:        
                # normalize all tokens by cls token's norm
                clip_text = clip_emb / torch.norm(clip_pooled.unsqueeze(1), dim=-1).reshape(-1, 1, 1)
            else:
                clip_text = nn.functional.normalize(clip_text, dim=-1)
        return clip_text

    # def org_embed_text(self, text_samples):
    #     clip_text = clip.tokenize(text_samples).to(self.device)
    #     clip_text = self.clip.encode_text(clip_text)
    #     if self.clamp_embs:
    #         clip_text = torch.clamp(clip_text, -1.5, 1.5)
    #     if self.norm_embs:
    #         clip_text = nn.functional.normalize(clip_text, dim=-1)
    #     return clip_text

    def embed_curated_annotations(self, annots):
        for i,b in enumerate(annots):
            t = ''
            while t == '':
                rand = torch.randint(5,(1,1))[0][0]
                t = b[0,rand]
            if i==0:
                txt = np.array(t)
            else:
                txt = np.vstack((txt,t))
        txt = txt.flatten()
        return self.embed_text(txt)

# for prior
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists
from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig
# vd prior
from dalle2_pytorch.dalle2_pytorch import RotaryEmbedding, CausalTransformer, SinusoidalPosEmb, MLP, Rearrange, repeat, rearrange, prob_mask_like, LayerNorm, RelPosBias, Attention, FeedForward

class BrainDiffusionPrior(DiffusionPrior):
    """ 
    Differences from original:
    - Allow for passing of generators to torch random functions
    - Option to include the voxel2clip model and pass voxels into forward method
    - Return predictions when computing loss
    - Load pretrained model from @nousr trained on LAION aesthetics
    """
    def __init__(self, *args, **kwargs):
        voxel2clip = kwargs.pop('voxel2clip', None)
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, self_cond = None, clip_denoised = True, cond_scale = 1.,
                generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = t, text_cond = text_cond, self_cond = self_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn_like(x)
            # noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps = None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, **kwargs, timesteps = timesteps)

        # print("PS removed all image_embed_scale instances!")
        image_embed = normalized_image_embed #/ self.image_embed_scale
        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale = 1., generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            image_embed = torch.randn(shape, device = device)
        else:
            image_embed = torch.randn(shape, device = device, generator=generator)
        x_start = None # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in reversed(range(0, self.noise_scheduler.num_timesteps)):
            times = torch.full((batch,), i, device = device, dtype = torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond = text_cond, self_cond = self_cond, cond_scale = cond_scale, 
                                                 generator=generator)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond = self_cond,
            text_cond_drop_prob = self.text_cond_drop_prob,
            image_cond_drop_prob = self.image_cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = nn.functional.mse_loss(pred, target) # mse
        # print("1", loss)
        # loss += (1 - nn.functional.cosine_similarity(pred, target).mean())
        # print("2", (1 - nn.functional.cosine_similarity(pred, target).mean()))
        return loss, pred

    def forward(
        self,
        text = None,
        image = None,
        voxel = None,
        text_embed = None,      # allow for training on preprocessed CLIP text and image embeddings
        image_embed = None,
        text_encodings = None,  # as well as CLIP text encodings
        *args,
        **kwargs
    ):
        assert exists(text) ^ exists(text_embed) ^ exists(voxel), 'either text, text embedding, or voxel must be supplied'
        assert exists(image) ^ exists(image_embed), 'either image or image embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        if exists(voxel):
            assert exists(self.voxel2clip), 'voxel2clip must be trained if you wish to pass in voxels'
            assert not exists(text_embed), 'cannot pass in both text and voxels'
            if self.voxel2clip.use_projector:
                clip_voxels_mse, clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse
            else:
                clip_voxels = self.voxel2clip(voxel)
                text_embed = clip_voxels_mse = clip_voxels
            # text_embed = self.voxel2clip(voxel)

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # PS: I dont think we need this? also if uncommented this does in-place global variable change
        # scale image embed (Katherine)
        # image_embed *= self.image_embed_scale

        # calculate forward loss

        loss, pred = self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)
        
        # undo the scaling so we can directly use it for real mse loss and reconstruction
        return loss, pred

class PriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps = None,
        num_time_embeds = 1,
        # num_image_embeds = 1,
        # num_brain_embeds = 1,
        num_tokens = 257,
        causal = True,
        learned_query_mode = 'none',
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        if self.learned_query_mode == 'token':
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if self.learned_query_mode == 'pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        if self.learned_query_mode == 'all_pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens*2+1, dim) * scale)
        self.causal_transformer = FlaggedCausalTransformer(dim = dim, causal=causal, **kwargs)

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        self.num_tokens = num_tokens
        self.self_cond = False

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, brain_cond_drop_prob = 1., image_cond_drop_prob = 1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        self_cond=None,
        brain_embed=None,
        text_embed=None,
        brain_cond_drop_prob = 0.,
        text_cond_drop_prob = None,
        image_cond_drop_prob = 0.
    ):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob
        # WE MAY WANT THESE
        # image_embed = image_embed.view(len(image_embed),-1,768)
        # brain_embed = brain_embed.view(len(brain_embed),-1,768)
        
        # text_embed = text_embed.view(len(text_embed),-1,768)
        # print(*image_embed.shape)  
        # for image : [257,768]   for txt : [1,768]
        # print(*image_embed.shape, image_embed.device, image_embed.dtype)
        # print("image_embed shape:", image_embed.shape)

        # if len(image_embed.shape) == 2:
        #     batch, dim = image_embed.shape
        #     device = image_embed.device
        #     dtype = image_embed.dtype
        # else:
        batch = text_embed.shape[0]
        _, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        # print(f'batch is {batch}')
        # num_time_embeds, num_image_embeds, num_brain_embeds = self.num_time_embeds, self.num_image_embeds, self.num_brain_embeds
        
        # classifier free guidance masks
        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device = device)
        brain_keep_mask = rearrange(brain_keep_mask, 'b -> b 1 1')

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device = device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

        # mask out brain embeddings with null brain embeddings

        # import pdb; pdb.set_trace()
        null_brain_embeds = self.null_brain_embeds.to(brain_embed.dtype)
        # print("brain_keep_mask shape:", brain_keep_mask.shape)
        # print("brain_embed shape:", brain_embed.shape)
        # print("null_brain_embeds shape:", null_brain_embeds.shape)
        brain_embed = torch.where(
            brain_keep_mask,
            brain_embed,
            null_brain_embeds[None]
        )

        # mask out image embeddings with null image embeddings
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        
        image_embed = torch.where(
            image_keep_mask,
            image_embed,
            null_image_embed[None]
        )

        # whether brain embedding is used for conditioning depends on whether brain encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right
        if self.continuous_embedded_time:
            # if continuous cast to flat, else keep int for indexing embeddings
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        if self.learned_query_mode == 'token':
            learned_queries = repeat(self.learned_query, 'n d -> b n d', b = batch)
        elif self.learned_query_mode == 'pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch)
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        elif self.learned_query_mode == 'all_pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b = batch)
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        
        tokens = torch.cat((
            brain_embed,  # 257
            time_embed,  # 1
            image_embed,  # 257
            learned_queries  # 257
        ), dim = -2)
        if self.learned_query_mode == 'all_pos_emb':
            tokens = tokens + pos_embs

        # attend
        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)
        pred_image_embed = tokens[..., -self.num_tokens:, :]

        return pred_image_embed

class FlaggedCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True,
        causal=True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)
    
#Subclass for GNET
class TrunkBlock(nn.Module):
    def __init__(self, feat_in, feat_out):
        super(TrunkBlock, self).__init__()
        self.conv1 = nn.Conv2d(feat_in, int(feat_out*1.), kernel_size=3, stride=1, padding=1, dilation=1)
        self.drop1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(feat_in, eps=1e-05, momentum=0.25, affine=True, track_running_stats=True)

        torch.nn.init.xavier_normal_(self.conv1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.constant_(self.conv1.bias, 0.0) # current
        
    def forward(self, x):
        return torch.nn.functional.relu(self.conv1(self.drop1(self.bn1(x))))

#Subclass for GNET
class PreFilter(nn.Module):
    def __init__(self):
        super(PreFilter, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )        
        
    def forward(self, x):
        c1 = self.conv1(x)
        y = self.conv2(c1)
        return y 

#Subclass for GNET
class EncStage(nn.Module):
    def __init__(self, trunk_width=64, pass_through=64):
        super(EncStage, self).__init__()
        self.conv3  = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=0)
        self.drop1  = nn.Dropout2d(p=0.5, inplace=False) ##
        self.bn1    = nn.BatchNorm2d(192, eps=1e-05, momentum=0.25, affine=True, track_running_stats=True) ##
        self.pool1  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ##
        self.tw = int(trunk_width)
        self.pt = int(pass_through)
        ss = (self.tw + self.pt)
        self.conv4a  = TrunkBlock(128, ss)
        self.conv5a  = TrunkBlock(ss, ss)
        self.conv6a  = TrunkBlock(ss, ss)
        self.conv4b  = TrunkBlock(ss, ss)
        self.conv5b  = TrunkBlock(ss, ss)
        self.conv6b  = TrunkBlock(ss, self.tw)
        ##
        torch.nn.init.xavier_normal_(self.conv3.weight, gain=torch.nn.init.calculate_gain('relu'))        
        torch.nn.init.constant_(self.conv3.bias, 0.0)
        
    def forward(self, x):
        c3 = (torch.nn.functional.relu(self.conv3(self.drop1(self.bn1(x))), inplace=False))
        c4a = self.conv4a(c3)
        c4b = self.conv4b(c4a)
        c5a = self.conv5a(self.pool1(c4b))
        c5b = self.conv5b(c5a)
        c6a = self.conv6a(c5b)
        c6b = self.conv6b(c6a)

        return [torch.cat([c3, c4a[:,:self.tw], c4b[:,:self.tw]], dim=1), 
                torch.cat([c5a[:,:self.tw], c5b[:,:self.tw], c6a[:,:self.tw], c6b], dim=1)], c6b
    
#Subclass for GNET
class GEncoder(nn.Module):
    def __init__(self, mu, trunk_width, pass_through=64 ):
        super(GEncoder, self).__init__()
        self.mu = nn.Parameter(torch.from_numpy(mu), requires_grad=False) #.to(device)
        self.pre = PreFilter()
        self.enc = EncStage(trunk_width, pass_through) 

    def forward(self, x):
        fmaps, h = self.enc(self.pre(x - self.mu))
        return x, fmaps, h

#Main GNET model class
class Torch_LayerwiseFWRF(nn.Module):
    def __init__(self, fmaps, nv=1, pre_nl=None, post_nl=None, dtype=np.float32):
        super(Torch_LayerwiseFWRF, self).__init__()
        self.fmaps_shapes = [list(f.size()) for f in fmaps]
        self.nf = np.sum([s[1] for s in self.fmaps_shapes])
        self.pre_nl  = pre_nl
        self.post_nl = post_nl
        self.nv = nv
        ##
        self.rfs = []
        self.sm = nn.Softmax(dim=1)
        for k,fm_rez in enumerate(self.fmaps_shapes):
            rf = nn.Parameter(torch.tensor(np.ones(shape=(self.nv, fm_rez[2], fm_rez[2]), dtype=dtype), requires_grad=True))
            self.register_parameter('rf%d'%k, rf)
            self.rfs += [rf,]
        self.w  = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(self.nv, self.nf)).astype(dtype=dtype), requires_grad=True))
        self.b  = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(self.nv,)).astype(dtype=dtype), requires_grad=True))
        
    def forward(self, fmaps):
        phi = []
        for fm,rf in zip(fmaps, self.rfs): #, self.scales):
            g = self.sm(torch.flatten(rf, start_dim=1))
            f = torch.flatten(fm, start_dim=2)  # *s
            if self.pre_nl is not None:          
                f = self.pre_nl(f)
            # fmaps : [batch, features, space]
            # v     : [nv, space]
            phi += [torch.tensordot(g, f, dims=[[1],[2]]),] # apply pooling field and add to list.
            # phi : [nv, batch, features] 
        Phi = torch.cat(phi, dim=2)
        if self.post_nl is not None:
            Phi = self.post_nl(Phi)
        vr = torch.squeeze(torch.bmm(Phi, torch.unsqueeze(self.w,2))).t() + torch.unsqueeze(self.b,0)
        return vr
    
class GNet8_Encoder():
    
    def __init__(self, subject = 1, device = "cuda", model_path = "gnet_multisubject.pt"):
        
        # Setting up Cuda
        self.device = torch.device(device)
        torch.backends.cudnn.enabled=True
        # Subject number
        self.subject = subject
        
        # Vector type
        self.vector = "images"
        
        # x size
        subject_sizes = [0, 15724, 14278, 15226, 13153, 13039, 17907, 12682, 14386]
        self.x_size = subject_sizes[self.subject]
        
        # Reload joined GNet model files
        self.joined_checkpoint = torch.load(model_path, map_location=self.device)
        
        self.subjects = list(self.joined_checkpoint['voxel_mask'].keys())
        self.gnet8j_voxel_mask = self.joined_checkpoint['voxel_mask']
        self.gnet8j_voxel_roi  = self.joined_checkpoint['voxel_roi']
        self.gnet8j_voxel_index= self.joined_checkpoint['voxel_index']
        self.gnet8j_brain_nii_shape= self.joined_checkpoint['brain_nii_shape']
        self.gnet8j_val_cc = self.joined_checkpoint['val_cc']
        
            
    
    def load_image(self, image_path):
        
        image = PIL.Image.open(image_path).convert('RGB')
        
        w, h = 227, 227  # resize to integer multiple of 64
        imagePil = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
        image = np.array(imagePil).astype(np.float32) / 255.0
        
        return image  
    
    # Rebuild Model
    def _model_fn(self, _ext, _con, _x):
        '''model consists of an extractor (_ext) and a connection model (_con)'''
        _y, _fm, _h = _ext(_x)
        return _con(_fm)

    def _pred_fn(self, _ext, _con, xb):
        return self._model_fn(_ext, _con, torch.from_numpy(xb).to(self.device))  
                    
    def subject_pred_pass(self, _pred_fn, _ext, _con, x, batch_size):
        pred = _pred_fn(_ext, _con, x[:batch_size]) # this is just to get the shape
        pred = np.zeros(shape=(len(x), pred.shape[1]), dtype=np.float32) # allocate
        for rb,_ in utils.iterate_range(0, len(x), batch_size):
            pred[rb] = utils.get_value(_pred_fn(_ext, _con, x[rb]))
        return pred

    def gnet8j_predictions(self, image_data, _pred_fn, trunk_width, pass_through, checkpoint, mask, batch_size, device=torch.device("cuda:0")):
        
        subjects = list(image_data.keys())

        if(mask is None):
            subject_nv = {s: len(v) for s,v in checkpoint['val_cc'].items()} 
        else:
            subject_nv = {s: len(v) for s,v in checkpoint['val_cc'].items()}    
            subject_nv[subjects[0]] = int(torch.sum(mask == True)) 

        # allocate
        subject_image_pred = {s: np.zeros(shape=(len(image_data[s]), subject_nv[s]), dtype=np.float32) for s in subjects}
        # print(subject_image_pred)
        _log_act_fn = lambda _x: torch.log(1 + torch.abs(_x))*torch.tanh(_x)
        
        best_params = checkpoint['best_params']
        # print(best_params)
        shared_model = GEncoder(np.array(checkpoint['input_mean']).astype(np.float32), trunk_width=trunk_width, pass_through=pass_through).to(device)
        shared_model.load_state_dict(best_params['enc'])
        shared_model.eval() 

        # example fmaps
        rec, fmaps, h = shared_model(torch.from_numpy(image_data[list(image_data.keys())[0]][:20]).to(device))                                     
        for s in subjects:
            sd = Torch_LayerwiseFWRF(fmaps, nv=subject_nv[s], pre_nl=_log_act_fn, post_nl=_log_act_fn, dtype=np.float32).to(device) 
            params = best_params['fwrfs'][s]
            
            if(mask is None):
                sd.load_state_dict(params)
            
            else:
                masked_params = {}
                for key, value in params.items():
                    masked_params[key] = value[mask]
                    
                sd.load_state_dict(masked_params)
                
            # print(params['w'].shape)
            # print(params['b'].shape)
            # sd.load_state_dict(best_params['fwrfs'][s])
            sd.eval() 
            # print(sd)
            
            subject_image_pred[s] = self.subject_pred_pass(_pred_fn, shared_model, sd, image_data[s], batch_size)

        return subject_image_pred

    def predict(self, images, mask = None):
        self.stim_data = {}
        data = []
        w, h = 227, 227  # resize to integer multiple of 64
        
        if(isinstance(images, list)):
            for i in range(len(images)):
                
                imagePil = images[i].convert("RGB").resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
                image = np.array(imagePil).astype(np.float32) / 255.0
                data.append(image)
            
        elif(isinstance(images, torch.Tensor)):
            for i in range(images.shape[0]):
                
                imagePil = utils.process_image(images[i], w, h)
                image = np.array(imagePil).astype(np.float32) / 255.0
                data.append(image)
            
        
        self.stim_data[self.subject] = np.moveaxis(np.array(data), 3, 1)

        gnet8j_image_pred = self.gnet8j_predictions(self.stim_data, self._pred_fn, 64, 192, self.joined_checkpoint, mask, batch_size=100, device=self.device)

        return torch.from_numpy(gnet8j_image_pred[self.subject])
    



from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from utils import retrieve_timesteps

class DiffusionPriorUNet(nn.Module):
    def __init__(
        self, 
        embed_dim=768,          # Updated from 1024 to 768
        cond_dim=42,
        hidden_dim=[768, 512, 256, 128, 64],
        time_embed_dim=512,
        act_fn=nn.SiLU,
        dropout=0.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # 1. Time Embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)

        # 3.1 Input Layer: Processes each token in the sequence
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )

        # 3.2 Hidden Encoder Layers
        self.num_layers = len(hidden_dim)
        self.encode_time_embedding = nn.ModuleList([
            TimestepEmbedding(time_embed_dim, hidden_dim[i]) 
            for i in range(self.num_layers - 1)
        ])
        self.encode_cond_embedding = nn.ModuleList([
            nn.Linear(cond_dim, hidden_dim[i]) 
            for i in range(self.num_layers - 1)
        ])
        self.encode_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                nn.LayerNorm(hidden_dim[i+1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in range(self.num_layers - 1)
        ])

        # 3.3 Hidden Decoder Layers
        self.decode_time_embedding = nn.ModuleList([
            TimestepEmbedding(time_embed_dim, hidden_dim[i]) 
            for i in reversed(range(1, self.num_layers))
        ])
        self.decode_cond_embedding = nn.ModuleList([
            nn.Linear(cond_dim, hidden_dim[i]) 
            for i in reversed(range(1, self.num_layers))
        ])
        self.decode_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i-1]),
                nn.LayerNorm(hidden_dim[i-1]),
                act_fn(),
                nn.Dropout(dropout),
            ) for i in reversed(range(1, self.num_layers))
        ])

        # 3.4 Output Layer: Maps back to embedding dimension
        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)
        
    def forward(self, x, t, c=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, embed_dim)
            t: Tensor of shape (batch_size,)
            c: Tensor of shape (batch_size, cond_dim) or None
        Returns:
            Tensor of shape (batch_size, seq_length, embed_dim)
        """
        batch_size, seq_length, _ = x.shape

        # 1. Time Embedding
        t_emb = self.time_proj(t)  # (batch_size, time_embed_dim)
        # Do NOT expand t_emb here; handle expansion within each layer

        # 3.1 Input Layer
        x = self.input_layer(x)  # (batch_size, seq_length, hidden_dim[0])

        # 3.2 Hidden Encoder
        hidden_activations = []
        for i in range(self.num_layers - 1):
            hidden_activations.append(x)
            
            # Get time embedding for current layer
            t_layer_emb = self.encode_time_embedding[i](t_emb)  # (batch_size, hidden_dim[i])
            t_layer_emb = t_layer_emb.unsqueeze(1).repeat(1, seq_length, 1)  # (batch_size, seq_length, hidden_dim[i])
            
            # Get conditional embedding for current layer, if provided
            if c is not None:
                c_layer_emb = self.encode_cond_embedding[i](c)  # (batch_size, hidden_dim[i])
                c_layer_emb = c_layer_emb.unsqueeze(1).repeat(1, seq_length, 1)  # (batch_size, seq_length, hidden_dim[i])
            else:
                # Create a tensor of zeros with the same shape as t_layer_emb
                c_layer_emb = torch.zeros_like(t_layer_emb, device=x.device)
            
            # Add time and conditional embeddings to x
            x = x + t_layer_emb + c_layer_emb  # (batch_size, seq_length, hidden_dim[i])
            
            # Pass through encoder layer
            x = self.encode_layers[i](x)  # (batch_size, seq_length, hidden_dim[i+1])
        
        # 3.3 Hidden Decoder
        for i in range(self.num_layers - 1):
            # Get decode time embedding for current layer
            t_layer_emb = self.decode_time_embedding[i](t_emb)  # (batch_size, hidden_dim[i])
            t_layer_emb = t_layer_emb.unsqueeze(1).repeat(1, seq_length, 1)  # (batch_size, seq_length, hidden_dim[i])
            
            # Get decode conditional embedding for current layer, if provided
            if c is not None:
                c_layer_emb = self.decode_cond_embedding[i](c)  # (batch_size, hidden_dim[i])
                c_layer_emb = c_layer_emb.unsqueeze(1).repeat(1, seq_length, 1)  # (batch_size, seq_length, hidden_dim[i])
            else:
                # Create a tensor of zeros with the same shape as t_layer_emb
                c_layer_emb = torch.zeros_like(t_layer_emb, device=x.device)
            
            # Add time and conditional embeddings to x
            x = x + t_layer_emb + c_layer_emb  # (batch_size, seq_length, hidden_dim[i])
            
            # Pass through decoder layer
            x = self.decode_layers[i](x)  # (batch_size, seq_length, hidden_dim[i-1])
            
            # Add skip connection from encoder
            x = x + hidden_activations[-1 - i]  # (batch_size, seq_length, hidden_dim[i-1])
        
        # 3.4 Output Layer
        x = self.output_layer(x)  # (batch_size, seq_length, embed_dim)

        return x

class DiffusionPrior:
    def __init__(self, prior_unet=None, scheduler=None, device='cuda'):
        self.prior_unet = prior_unet.to(device)
        
        if scheduler is None:
            self.scheduler = DDPMScheduler() 
        else:
            self.scheduler = scheduler
            
        self.device = device
        
    def train(self, dataloader, num_epochs=10, learning_rate=1e-4, wandb_log=False):
        self.prior_unet.train()
        device = self.device
        criterion = nn.MSELoss(reduction='mean')  # Changed to 'mean' for simplicity
        # Add weight decay in the  future here
        optimizer = optim.Adam(self.prior_unet.parameters(), lr=learning_rate)
        
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(dataloader) * num_epochs),
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps
        print(f'Number of training timesteps: {num_train_timesteps}')
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            loss_sum = 0
            for batch in dataloader:
                c_embeds = batch['clip_pred'].to(device).to(torch.float32)
                h_embeds = batch['clip_target'].to(device).to(torch.float32)  # (batch_size, 257, 768)
                N = h_embeds.shape[0]
                print(f'Batch size: {N}')
                # 1. Randomly replace c_embeds with None
                if torch.rand(1).item() < 0.1:
                    c_embeds = None

                # 2. Generate noisy embeddings as input
                noise = torch.randn_like(h_embeds)

                # 3. Sample timestep
                timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

                # 4. Add noise to h_embedding
                perturbed_h_embeds = self.scheduler.add_noise(
                    h_embeds,
                    noise,
                    timesteps
                ).to(torch.float32)  # (batch_size, 257, 768)

                # 5. Predict noise
                noise_pred = self.prior_unet(perturbed_h_embeds, timesteps.to(torch.float32), c_embeds)
                
                # 6. Compute loss
                loss = criterion(noise_pred, noise)  # Scalar
                
                # 7. Backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.prior_unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                loss_sum += loss.item()
                if wandb_log:
                    wandb.log({'train/loss': loss.item()})

            loss_epoch = loss_sum / len(dataloader)
            tqdm.write(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss_epoch:.4f}')

    def generate(
        self, 
        c_embeds=None, 
        num_inference_steps=50, 
        timesteps=None,
        guidance_scale=5.0,
        generator=None
    ):
        """
        Args:
            c_embeds: Tensor of shape (batch_size, cond_dim) or None
            num_inference_steps: Number of diffusion steps
            timesteps: Optional custom timesteps
            guidance_scale: Scale for classifier-free guidance
            generator: Torch generator for reproducibility
        Returns:
            Generated embeddings of shape (batch_size, 257, 768)
        """
        self.prior_unet.eval()
        N = c_embeds.shape[0] if c_embeds is not None else 1

        # 1. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device, timesteps
        )

        # 2. Prepare c_embeds
        if c_embeds is not None:
            c_embeds = c_embeds.to(self.device)

        # 3. Initialize noisy embeddings: (N, 257, 768)
        h_t = torch.randn(N, 257, self.prior_unet.embed_dim, generator=generator, device=self.device)

        # 4. Denoising loop
        for step_index, t in enumerate(timesteps):
            t_tensor = torch.full((N,), t, device=self.device, dtype=torch.long)  # (batch_size,)
            
            # 4.1 Predict noise
            if guidance_scale == 0 or c_embeds is None:
                noise_pred = self.prior_unet(h_t, t_tensor, c_embeds)
            else:
                # Predict noise with and without condition
                noise_pred_cond = self.prior_unet(h_t, t_tensor, c_embeds)
                noise_pred_uncond = self.prior_unet(h_t, t_tensor, None)
                # Apply classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 4.2 Compute previous noisy sample
            h_t = self.scheduler.step(noise_pred, t, h_t, generator=generator).prev_sample
        
        return h_t
