#!/usr/bin/env bash
set -euo pipefail

# Kaggle setup: installs sam-audio and its dependencies.
# Kaggle provides PyTorch + CUDA pre-installed.
#
# Usage in a Kaggle notebook cell:
#   !git clone https://github.com/YOUR_USER/samu.git
#   %cd samu
#   !bash kaggle_setup.sh

export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"

# Save Kaggle's pre-installed torch version so we can restore it if pip upgrades it
TORCH_VER=$(python -c "import torch; print(torch.__version__)")

echo "==> Step 1/3: Installing sam-audio with all dependencies..."
pip install -q git+https://github.com/facebookresearch/sam-audio.git

echo "==> Step 2/3: Fixing version conflicts..."
# Downgrade huggingface-hub (sam-audio needs <1.0 API) and transformers (needs matching hf-hub)
pip install -q "huggingface-hub>=0.20.0,<1.0.0" "transformers>=4.54.0,<5.0.0"

# Restore Kaggle's torch/torchaudio if they got upgraded (they must match)
pip install -q "torch==$TORCH_VER" "torchaudio" --force-reinstall --no-deps 2>/dev/null || true

echo "==> Step 3/3: Installing project dependencies..."
pip install -q python-dotenv

echo "==> Done. Run: python main.py /kaggle/input/YOUR_DATASET/ -o /kaggle/working/output -v"
