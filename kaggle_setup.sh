#!/usr/bin/env bash
set -euo pipefail

# Kaggle setup: installs sam-audio and its dependencies.
# Kaggle provides PyTorch + CUDA. We pin compatible versions first,
# then install everything with --no-deps to avoid version conflicts
# with Kaggle's pre-installed packages.
#
# Usage in a Kaggle notebook cell:
#   !git clone https://github.com/YOUR_USER/samu.git
#   %cd samu
#   !bash kaggle_setup.sh

export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"

echo "==> Step 1/3: Pinning compatible core packages..."
pip install -q \
    "huggingface-hub>=0.20.0,<1.0.0" \
    "transformers>=4.54.0,<5.0.0" \
    "protobuf>=3.20.3,<4.0" \
    "numpy>=1.24.0,<2.0"

echo "==> Step 2/3: Installing sam-audio (no-deps to avoid conflicts)..."
pip install -q --no-deps git+https://github.com/facebookresearch/sam-audio.git
pip install -q --no-deps git+https://github.com/facebookresearch/dacvae.git
pip install -q --no-deps git+https://github.com/facebookresearch/ImageBind.git
pip install -q --no-deps git+https://github.com/lematt1991/CLAP.git
pip install -q --no-deps "git+https://github.com/facebookresearch/perception_models@unpin-deps"

echo "==> Step 3/3: Installing remaining dependencies..."
pip install -q --no-deps descript-audiotools argbind
pip install -q xformers python-dotenv soundfile einops safetensors torchdiffeq \
    sentencepiece ftfy regex scipy scikit-learn timm pytorchvideo pydub

echo "==> Done. Run: python main.py /kaggle/input/YOUR_DATASET/ -o /kaggle/working/output -v"
