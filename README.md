# samu

Isolate bird sounds (or any target sound) from audio files using [Meta's SAM-Audio](https://github.com/facebookresearch/sam-audio) model.

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-compatible GPU recommended (CPU works but is much slower)
- FFmpeg installed on your system path

## Setup

### 1. Authenticate with HuggingFace

SAM-Audio model checkpoints are gated. You need to:

1. Request access at [facebook/sam-audio-base](https://huggingface.co/facebook/sam-audio-base) (or the small/large variant you plan to use)
2. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set up your token:

```bash
cp .env.sample .env
# Edit .env and paste your token
```

Or authenticate via the CLI instead:

```bash
huggingface-cli login
```

### 2. Install dependencies

```bash
uv sync
```

## Usage

### CLI

Separate bird sounds from a single file:

```bash
uv run python main.py recording.wav
```

Process a directory of audio files:

```bash
uv run python main.py ./recordings/ -o ./clean_birds/
```

Custom text prompt and model size:

```bash
uv run python main.py recording.wav -d "bird singing" -m large -o ./output/
```

Skip saving the residual (non-bird) audio:

```bash
uv run python main.py recording.wav --no-residual
```

Higher quality with more reranking candidates (slower):

```bash
uv run python main.py recording.wav --reranking-candidates 8
```

All options:

```
usage: main.py [-h] [-o OUTPUT_DIR] [-d DESCRIPTION] [-m {small,base,large}]
            [--device DEVICE] [--no-residual]
            [--predict-spans | --no-predict-spans]
            [--reranking-candidates N] [-v]
            input

positional arguments:
  input                 Path to an audio file or a directory of audio files

options:
  -o, --output-dir      Directory for separated audio (default: ./output)
  -d, --description     Text prompt for target sound (default: "bird singing")
  -m, --model           Model size: small, base, large (default: base)
  --device              Force device: cuda, cpu (default: auto-detect)
  --no-residual         Skip saving the residual audio
  --predict-spans       Enable span prediction (default: on)
  --no-predict-spans    Disable span prediction
  --reranking-candidates  Number of candidates for reranking (default: 1)
  -v, --verbose         Enable verbose logging
```

### Python API

```python
from separator import AudioSeparator

sep = AudioSeparator(model_size="base")

# Separate a single file
result = sep.separate("recording.wav", description="bird singing")
# result.target   -> torch.Tensor of isolated bird sounds
# result.residual -> torch.Tensor of everything else

# Process and save to disk
sep.process_file("recording.wav", output_dir="output/", description="bird singing")
```

## Output

For each input file, two files are written to the output directory:

- `<name>_target.wav` -- the isolated target sound (e.g. birds)
- `<name>_residual.wav` -- everything else (noise, other animals, etc.)

## Models

| Model | Quality | Speed | VRAM |
|-------|---------|-------|------|
| `small` | Good | Fastest | ~2 GB |
| `base` | Better | Balanced | ~3 GB |
| `large` | Best | Slowest | ~6 GB |

VRAM estimates are with bfloat16 precision on CUDA. CPU mode uses float32 and requires no GPU memory.
