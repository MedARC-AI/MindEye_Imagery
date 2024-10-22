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

# class ClipTokenWeightEncoder:
#     def encode_token_weights(self, token_weight_pairs):
#         bs = len(token_weight_pairs)
#         # tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
#         start = time.time()
#         out, pooled = self(token_weight_pairs)
#         print(f"cliptokenweightencoder encoder time {time.time() - start}")
#         print(f"cliptokenweightencoder out shape: {out.shape}")
#         if pooled is not None:
#             first_pooled = pooled[0:bs].cpu()
#             print(f"cliptokenweightencoder first_pooled shape: {first_pooled.shape}")
#         else:
#             first_pooled = pooled
#         output = out[0:bs]
#         print(f"cliptokenweightencoder: {output.shape}, {first_pooled}")
#         return output.cpu(), first_pooled


class SD35_Reconstructor(object):
    @torch.no_grad()
    def __init__(self, device="cuda", cache_dir="../cache/", embedder_only=False):
        print(f"Stable Diffusion 3.5 Reconstructor: Loading model...")
        self.device = device
        self.cache_dir = cache_dir
        self.dtype = torch.bfloat16
        self.verbose=False
        self.tokenizer = SD3Tokenizer()
        print("Loading OpenAI CLIP L...")
        self.clip_l = ClipL(cache_dir, device=device)
        print("Loading OpenCLIP bigG...")
        self.clip_g = ClipG(cache_dir, device=device)
        print("Loading Google T5-v1-XXL...")
        self.t5xxl = T5XXL(cache_dir, device=device)
        if not embedder_only:
            print("Loading VAE model...")
            self.vae = VAE(f"{cache_dir}/sd3.5_large.safetensors")
            print(f"Loading SD3 model...")
            self.sd3 = SD3(f"{cache_dir}/sd3.5_large.safetensors", shift=3.0, verbose=False)
            
        print("Models loaded.")
    
    # @torch.no_grad()
    # def embed_text(self, text):
    #     if isinstance(text, list):
    #         tokens = {"l" : [], "g" : [], "t5xxl" : []}
    #         for t in text:
    #             s_tokens = self.tokenizer.tokenize_with_weights(t)
    #             tokens["l"].append(list(map(lambda a: a[0], s_tokens["l"][0])))
    #             tokens["g"].append(list(map(lambda a: a[0], s_tokens["g"][0])))
    #             tokens["t5xxl"].append(list(map(lambda a: a[0], s_tokens["t5xxl"][0])))
    #     else:
    #         tokens = self.tokenizer.tokenize_with_weights(text)
    #     l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
    #     start = time.time()
    #     g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
    #     print(f"CLIP G took {time.time() - start} seconds")
    #     start = time.time()
    #     print("token length t5 ", len(tokens["t5xxl"][0]))
    #     t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
    #     print(f"T5 took {time.time() - start} seconds")
    #     lg_out = torch.cat([l_out, g_out], dim=-1)
    #     lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
    #     return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
    #         (l_pooled, g_pooled), dim=-1
    #     )
    
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
    def embed_latent(self, image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images)
        image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch).cpu()
        self.vae.model = self.vae.model.cpu()
        return latent
    
    @torch.no_grad()
    def reconstruct(self,
                    image=None, 
                    c_t=None, 
                    n_samples=1, 
                    strength=1.0, 
                    seed=None,
                    num_steps = 40,
                    cfg=4):
        height, width = 1024, 1024
        if strength < 1.0: # Prepare partially noised latents
            assert image is not None, "Image must be provided for strength < 1.0"
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image).resize((height,width))
            elif isinstance(image, PIL.Image.Image):
                image = image.resize((width, height), Image.LANCZOS)
            latent = self.embed_latent(image)
            latent = SD3LatentFormat().process_in(latent)
            
        else:
            latent = self.get_empty_latent(width, height)
            
        if int(strength * num_steps) > 0:
            neg_cond = self.embed_text("")
            if seed==None:
                seed = torch.randint(0, 100000, (1,)).item()
            sampled_latent = self.do_sampling(
                latent,
                seed,
                c_t,
                neg_cond,
                num_steps,
                cfg,
                "dpmpp_2m",
                strength,
            )
        else:
            sampled_latent = latent
            
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
        self.sd3.model = self.sd3.model.cuda()
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
        self.sd3.model = self.sd3.model.cpu()
        self.print("Sampling done")
        return latent
    
    def vae_decode(self, latent) -> Image.Image:
        self.print("Decoding latent to image...")
        latent = latent.cuda()
        self.vae.model = self.vae.model.cuda()
        image = self.vae.model.decode(latent)
        image = image.float()
        self.vae.model = self.vae.model.cpu()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        self.print("Decoded")
        return out_image
    
    def print(self, txt):
        if self.verbose:
            print(txt)