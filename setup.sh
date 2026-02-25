#!/usr/bin/env bash
set -euo pipefail

# SAM-Audio has heavy native dependencies that must be built from source.
# Installing them one at a time prevents OOM kills during compilation.

export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1

echo "==> Step 1/6: Syncing lightweight dependencies..."
uv sync

echo "==> Step 2/6: Installing ImageBind..."
uv pip install git+https://github.com/facebookresearch/ImageBind.git

echo "==> Step 3/6: Installing dacvae..."
uv pip install git+https://github.com/facebookresearch/dacvae.git

echo "==> Step 4/6: Installing CLAP..."
uv pip install git+https://github.com/lematt1991/CLAP.git

echo "==> Step 5/6: Installing perception_models..."
uv pip install "git+https://github.com/facebookresearch/perception_models@unpin-deps"

echo "==> Step 6/6: Installing sam-audio..."
uv pip install git+https://github.com/facebookresearch/sam-audio.git

echo "==> Done. Run 'uv run samu --help' to verify."
