# MIRAGE

This is the main working branch of MIRAGE. It uses the Stable Cascade diffusion model for reconstructions, and a set of Ridge regression models as the decoding backbone. 

To install the proper environment, follow `src/setup.sh`.

To use this branch, you must also clone the StableCascade repo from `https://github.com/Stability-AI/StableCascade.git` into your src directory, such that it is located at `MIRAGE/src/StableCascade/

You will need a checkpoint for the VDVAE to use the low level pipeline, you can download the checkpoint using the following command:
- `wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th`

It should download the Stable Cascade models you need automatically, but if it doesn't, you can also download the following files from the [stable cascade huggingface repo](https://huggingface.co/stabilityai/stable-cascade/tree/main), and place them into your `cache_dir`.
- effnet_encoder.safetensors
- previewer.safetensors
- stage_a.safetensors
- stage_b.safetensors
- stage_c.safetensors


## Installation

1. Git clone this repository:

```
git clone https://github.com/MedARC-AI/MIRAGE.git
```

2. Download necessary project files from the two hugginface repositories and place them in the same folder as your git clone.
    - REMOVED FOR ANONYMITY
    
Warning: **This will download over 300 GB of data!** You may want to only download some parts of the huggingface dataset (e.g., not all the pretrained models contained in "train_logs", only one of the preparations of brain activity—whole brain or not—whole brain betas are only necessary for SNR thresholding.)


3. Run ```. src/setup.sh``` to install a new "mei-env" virtual environment. Make sure the virtual environment is activated with "source mirage-env/bin/activate".

## Usage

- ```src/Train.ipynb``` trains models using our ridge regression backbone
- ```src/recon_inference_mi.ipynb``` will run inference on the NSD Imagery dataset using a trained model, outputting tensors of reconstructions/predicted captions/etc.
- ```src/final_evaluations_multi_mi.ipynb``` will compute quantitative metrics
