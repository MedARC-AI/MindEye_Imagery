#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

pip install --upgrade pip

python -m venv mei-env
source mei-env/bin/activate

git clone https://github.com/Stability-AI/StableCascade

pip install numpy matplotlib==3.8.2 jupyter jupyterlab_nvdashboard jupyterlab tqdm scikit-image==0.22.0 accelerate==0.24.1 webdataset==0.2.73 pandas==2.2.0 einops>=0.7.0 ftfy regex kornia==0.7.1 h5py==3.10.0 open_clip_torch torchvision==0.16.0 torch==2.1.0 transformers==4.37.2 xformers torchmetrics==1.3.0.post0 diffusers==0.30.3 deepspeed==0.13.1 wandb omegaconf==2.3.0 pytorch-lightning==2.0.1 sentence-transformers==2.5.1 evaluate==0.4.1 nltk==3.8.1 rouge_score==0.1.2 umap==0.1.1 nibabel==5.2.1 insightface==0.7.3 opencv-python==4.8.1.78 munch==4.0.0 onnxruntime==1.16.3 onnx2torch==1.5.13 pandas
pip install git+https://github.com/openai/CLIP.git --no-deps
pip install dalle2-pytorch

pip install git+https://github.com/black-forest-labs/flux.git#egg=flux[all]