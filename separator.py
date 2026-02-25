from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
import torchaudio
import torchaudio.transforms as T

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

MODEL_REPO_MAP = {
    "small": "facebook/sam-audio-small",
    "base": "facebook/sam-audio-base",
    "large": "facebook/sam-audio-large",
}


@dataclass
class SeparationResult:
    target: torch.Tensor
    residual: torch.Tensor
    sample_rate: int


def detect_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is unreliable for the attention ops SAM-Audio uses
    return torch.device("cpu")


def select_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class AudioSeparator:
    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        from sam_audio import SAMAudio, SAMAudioProcessor

        repo = MODEL_REPO_MAP[model_size]
        self.device = detect_device(device)
        self.dtype = dtype or select_dtype(self.device)

        log.info("Loading model %s on %s (%s)...", repo, self.device, self.dtype)
        self.model = SAMAudio.from_pretrained(repo)
        if self.dtype != torch.float32:
            self.model = self.model.to(self.dtype)
        self.model = self.model.to(self.device).eval()

        self.processor = SAMAudioProcessor.from_pretrained(repo)
        self.sample_rate: int = self.processor.audio_sampling_rate

    def _load_audio(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = T.Resample(sr, self.sample_rate)(wav)
        return wav.squeeze(0)

    def separate(
        self,
        input_path: str | Path,
        description: str = "bird singing",
        predict_spans: bool = True,
        reranking_candidates: int = 1,
    ) -> SeparationResult:
        input_path = Path(input_path)
        wav = self._load_audio(input_path)

        inputs = self.processor(
            audios=[wav.numpy()],
            descriptions=[description],
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)

        with torch.inference_mode():
            out = self.model.separate(
                inputs,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates,
            )

        target = out.target[0].float().cpu()
        min_len = min(wav.shape[-1], target.shape[-1])
        target = target[..., :min_len]
        original = wav[..., :min_len]
        residual = original - target

        return SeparationResult(
            target=target,
            residual=residual,
            sample_rate=self.sample_rate,
        )

    def process_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        description: str = "bird singing",
        save_residual: bool = True,
        predict_spans: bool = True,
        reranking_candidates: int = 1,
    ) -> Path:
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info("Processing: %s", input_path.name)
        result = self.separate(
            input_path,
            description=description,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates,
        )

        stem = input_path.stem
        target_path = output_dir / f"{stem}_target.wav"
        torchaudio.save(
            str(target_path),
            result.target.unsqueeze(0),
            result.sample_rate,
        )
        log.info("  Saved target: %s", target_path)

        if save_residual:
            residual_path = output_dir / f"{stem}_residual.wav"
            torchaudio.save(
                str(residual_path),
                result.residual.unsqueeze(0),
                result.sample_rate,
            )
            log.info("  Saved residual: %s", residual_path)

        return target_path
