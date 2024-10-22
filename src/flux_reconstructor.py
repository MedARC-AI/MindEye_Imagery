import os
import re
import time
from glob import iglob
from io import BytesIO

import streamlit as st
import torch
from einops import rearrange, repeat
from fire import Fire
from PIL import ExifTags, Image
import PIL
from st_keyup import st_keyup
from torchvision import transforms
from transformers import pipeline

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    configs,
    embed_watermark,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)
def model_summary_per_device(model):
    total_params = 0
    device_params = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        device = param.device
        
        if device not in device_params:
            device_params[device] = 0
        device_params[device] += num_params
        
    print(f"\nTotal Parameters: {total_params}")
    print("Total Parameters per Device:")
    for device, params in device_params.items():
        print(f"  {device}: {params}")

class Flux_Reconstructor(object):
    def __init__(self, device="cuda:0", cache_dir="../cache/", embedder_only=False, offload=False, max_length=64):
        print(f"Flux Reconstructor: Loading model...")
        self.device = device
        self.cache_dir = cache_dir
        self.dtype = torch.bfloat16
        self.offload = offload
        os.environ['HF_HOME'] = cache_dir
        
        if embedder_only:
            self.t5 = load_t5(device, max_length=max_length)
            self.clip = load_clip(device)
        else:
            self.model = load_flow_model("flux-dev", device="cpu" if offload else device)
            self.ae = load_ae("flux-dev", device="cpu" if offload else device)
        
        self.prep_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )
        
    @torch.inference_mode()
    def embed_text(self, text):
        clip_text = self.clip(text)
        t5_text = self.t5(text)
        return clip_text, t5_text 
    
    # @torch.inference_mode()
    # def embed_latent(self, images):
    #     latents = self.ae.encode(images.to(self.device))
    #     return latents

    @torch.inference_mode()
    def reconstruct(self,
                    image=None, 
                    c_t=None, 
                    t5=None, 
                    n_samples=1, 
                    strength=1.0, 
                    seed=None,
                    num_steps = 50,
                    cfg=3.5):
        height, width = 1024, 1024
        if seed==None:
            rng = torch.Generator(device="cpu")
            seed = rng.seed()
        x = get_noise(
            n_samples,
            height,
            width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=seed,
        )
        timesteps = get_schedule(
            num_steps,
            (x.shape[-1] * x.shape[-2]) // 4,
            shift=True,
        )
        if strength < 1.0: # Prepare partially noised latents
            if image is not None:
                if isinstance(image, Image.Image):
                    image = self.prep_transform(image)[None, ...]
                init_image = torch.nn.functional.interpolate(image, (height, width))
                if self.offload:
                    self.ae.to(self.device)
                
                init_image = self.ae.encode(init_image.to(self.device))
                
                if self.offload:
                    self.ae = self.ae.cpu()
                    torch.cuda.empty_cache()
                
                t_idx = int((1 - strength) * num_steps)
                t = timesteps[t_idx]
                timesteps = timesteps[t_idx:]
                x = t * x + (1.0 - t) * init_image.expand(n_samples, -1, -1, -1).to(x.dtype)
            else:
                raise ValueError("Image must be provided for strength < 1.0")
        elif strength == 0.0:
            return image
        
        # Prepare latents
        bs, c, h, w = x.shape
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if bs == 1 and n_samples > 1:
            img = repeat(img, "1 ... -> bs ...", bs=n_samples)
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=n_samples)
        
        #Prepare guidance
        t5 = t5.reshape((-1, 64, 4096)).expand(n_samples, -1, -1).to(torch.bfloat16)
        c_t = c_t.reshape((-1, 768)).expand(n_samples, -1).to(torch.bfloat16)
        t5_ids = torch.zeros(n_samples, t5.shape[1], 3)
        inp = {
            "img": img,
            "img_ids": img_ids.to(img.device),
            "txt": t5.to(img.device),
            "txt_ids": t5_ids.to(img.device),
            "vec": c_t.to(img.device),
        }
        if self.offload:
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)
        x = denoise(self.model, **inp, timesteps=timesteps, guidance=cfg)
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(self.device)
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.ae.decode(x)
        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()
        x = x.clamp(-1, 1)
        x = rearrange(x, "b c h w -> b h w c")
        img_list = []
        for img in x:
            img_list.append(Image.fromarray((127.5 * (img + 1.0)).cpu().byte().numpy()))
        if len(img_list) == 1:
            return img_list[0]
        return img_list