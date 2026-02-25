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

echo "==> Installing sam-audio and dependencies..."
pip install -q git+https://github.com/facebookresearch/sam-audio.git

echo "==> Installing project dependencies..."
pip install -q python-dotenv soundfile

echo "==> Fixing version conflicts with Kaggle environment..."
pip install -q "huggingface-hub>=0.20.0,<1.0.0" "protobuf>=3.20.3" "numpy>=1.24.0,<2.0"

echo "==> Done. Run: python main.py /kaggle/input/YOUR_DATASET/ -o /kaggle/working/output -v"
