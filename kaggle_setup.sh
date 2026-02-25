#!/usr/bin/env bash
set -euo pipefail

# Kaggle setup: installs sam-audio and its dependencies.
# Kaggle already provides PyTorch + CUDA, so this is straightforward.
#
# Usage in a Kaggle notebook cell:
#   !git clone https://github.com/YOUR_USER/samu.git
#   %cd samu
#   !bash kaggle_setup.sh

echo "==> Installing sam-audio and dependencies..."
pip install -q git+https://github.com/facebookresearch/sam-audio.git

echo "==> Installing project dependencies..."
pip install -q python-dotenv soundfile

echo "==> Done. Run: python main.py /kaggle/input/YOUR_DATASET/ -o /kaggle/working/output -v"
