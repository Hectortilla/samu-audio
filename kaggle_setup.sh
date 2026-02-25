#!/usr/bin/env bash
set -euo pipefail

# Kaggle setup: installs sam-audio and its dependencies.
# Kaggle already provides PyTorch + CUDA, so this is straightforward.
#
# Usage in a Kaggle notebook cell:
#   !git clone https://github.com/YOUR_USER/samu.git
#   %cd samu
#   !bash kaggle_setup.sh

# Auto-accept GitHub SSH host key (Kaggle doesn't have it in known_hosts)
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"

echo "==> Pinning compatible package versions..."
pip install -q "huggingface-hub>=0.20.0,<1.0.0" "transformers>=4.54.0,<5.0.0" \
    "protobuf>=3.20.3" "numpy>=1.24.0,<2.0"

echo "==> Installing sam-audio and dependencies..."
pip install -q --no-deps git+https://github.com/facebookresearch/sam-audio.git

echo "==> Installing sam-audio's transitive dependencies..."
pip install -q git+https://github.com/facebookresearch/dacvae.git \
    git+https://github.com/facebookresearch/ImageBind.git \
    git+https://github.com/lematt1991/CLAP.git \
    "git+https://github.com/facebookresearch/perception_models@unpin-deps"

echo "==> Installing project dependencies..."
pip install -q python-dotenv soundfile einops safetensors torchdiffeq \
    sentencepiece ftfy regex scipy scikit-learn audiobox_aesthetics

echo "==> Done. Run: python main.py /kaggle/input/YOUR_DATASET/ -o /kaggle/working/output -v"
