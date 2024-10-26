import os
import sys
from PIL import Image
import PIL
import numpy as np
import torch
import time
import yaml
from safetensors import safe_open
from tqdm import tqdm
from torchvision import transforms
import math
from sd35_models import SDVAE, BaseModel, CFGDenoiser, SD3LatentFormat, SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel, VAE, SD3, T5XXL, ClipL, ClipG, load_into, sample_dpmpp_2m
import sd35_models


class SD35_Reconstructor(object):
    @torch.no_grad()
    def __init__(self, device="cuda", cache_dir="../cache/", embedder_only=False):
        print(f"Stable Diffusion 3.5 Reconstructor: Loading model...")
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.verbose=False
        self.tokenizer = SD3Tokenizer()
        print("Loading OpenAI CLIP L...")
        self.clip_l = ClipL(cache_dir)
        print("Loading OpenCLIP bigG...")
        self.clip_g = ClipG(cache_dir)
        print("Loading Google T5-v1-XXL...")
        self.t5xxl = T5XXL(cache_dir)
        self.t5xxl.model = self.t5xxl.model.cuda()
        print("Loading VAE model...")
        self.vae = VAE(f"{cache_dir}/sd3.5_large.safetensors")
        self.vae.model = self.vae.model.cuda()
        if not embedder_only:
            
            print(f"Loading SD3 model...")
            self.sd3 = SD3(f"{cache_dir}/sd3.5_large.safetensors", shift=3.0, verbose=False)
            self.sd3.model = self.sd3.model.cuda()
        else:
            # These can only fit on the GPU if the main model is not loaded, they speed up feature extraction, but not necessary for reconstruction
            self.clip_l.model = self.clip_l.model.cuda()
            self.clip_g.model = self.clip_g.model.cuda()
            
        print("Models loaded.")
    
    @torch.no_grad()
    def embed_text(self, text):
        if isinstance(text, list):
            text = text[0]
        tokens = self.tokenizer.tokenize_with_weights(text)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )
    
    #This is offloading the model each time, probably want to remove
    @torch.no_grad()
    def embed_latent(self, image, height=1024, width=1024) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        image = image.convert("RGB")
        image = image.resize((height, width), Image.LANCZOS)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images)
        image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        latent = self.vae.model.encode(image_torch)
        return latent
    
    @torch.no_grad()
    def reconstruct(self,
                    image=None, 
                    latent=None,
                    c_t=None, 
                    t5=None,
                    n_samples=1, 
                    strength=1.0, 
                    seed=None,
                    num_steps = 40,
                    cfg=4,
                    height=1024,
                    width=1024):
        if strength < 1.0: # Prepare partially noised latents
            if latent is not None:
                latent = latent.reshape(1, 16, height//8, width//8).to(torch.float32)
            elif image is not None:
                if isinstance(image, torch.Tensor):
                    image = transforms.ToPILImage()(image).resize((height,width))
                elif isinstance(image, PIL.Image.Image):
                    image = image.resize((width, height), Image.LANCZOS)
                latent = self.embed_latent(image).to(torch.float32)
            else:
                raise ValueError("Image or latent must be provided for strength < 1.0")
            latent = SD3LatentFormat().process_in(latent)#.expand(n_samples, -1, -1, -1)
            
        else:
            latent = self.get_empty_latent(width, height)#.expand(n_samples, -1, -1, -1)
            
        if int(strength * num_steps) > 0:
            neg_cond = self.embed_text("") #.expand(n_samples, -1, -1) .expand(n_samples, -1)
            cond = (c_t.reshape((1,154,4096)).to(self.device, torch.float32), t5.reshape((1,2048)).to(self.device, torch.float32)) #these are actually flipped because I labeled them wrong (sorry)
            if seed==None:
                seed = torch.randint(0, 100000, (1,)).item()
            sampled_latent = self.do_sampling(
                latent,
                seed, # 
                cond,
                neg_cond,
                num_steps,
                cfg,
                "dpmpp_2m",
                strength,
            )
        else:
            sampled_latent = latent.to(torch.float16)
            
        image = self.vae_decode(sampled_latent)
        
        return image
    
        
    def get_empty_latent(self, width, height):
        self.print("Prep an empty latent...")
        return torch.ones(1, 16, height // 8, width // 8, device="cpu") * 0.0609

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        self.print(
            f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}"
        )
        return torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(latent.dtype)
    
    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}
    
    def do_sampling(
        self,
        latent,
        seed,
        conditioning,
        neg_cond,
        steps,
        cfg_scale,
        sampler="dpmpp_2m",
        denoise=1.0,
    ) -> torch.Tensor:
        self.print("Sampling...")
        latent = latent.half().cuda()
        # self.sd3.model = self.sd3.model.cuda()
        noise = self.get_noise(seed, latent).cuda()
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()
        sigmas = sigmas[int(steps * (1 - denoise)) :]
        conditioning = self.fix_cond(conditioning)
        neg_cond = self.fix_cond(neg_cond)
        extra_args = {"cond": conditioning, "uncond": neg_cond, "cond_scale": cfg_scale}
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise, latent, self.max_denoise(sigmas)
        )
        latent = sample_dpmpp_2m(
            CFGDenoiser(self.sd3.model), noise_scaled, sigmas, extra_args=extra_args
        )
        latent = SD3LatentFormat().process_out(latent)
        # self.sd3.model = self.sd3.model.cpu()
        self.print("Sampling done")
        return latent
    
    def vae_decode(self, latent) -> Image.Image:
        self.print("Decoding latent to image...")
        latent = latent.to(device)
        # self.vae.model = self.vae.model.cuda()
        image = self.vae.model.decode(latent)
        image = image.float()
        # self.vae.model = self.vae.model.cpu()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        self.print("Decoded")
        return out_image
    
    def print(self, txt):
        if self.verbose:
            print(txt)