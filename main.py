from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from separator import AUDIO_EXTENSIONS, AudioSeparator


def collect_audio_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not files:
            logging.error("No audio files found in %s", input_path)
            sys.exit(1)
        return files
    logging.error("Input path does not exist: %s", input_path)
    sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="samu",
        description="Isolate specific sounds from audio using Meta's SAM-Audio model.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to an audio file or a directory of audio files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for separated audio files (default: ./output)",
    )
    parser.add_argument(
        "-d", "--description",
        default="bird singing",
        help='Text prompt describing the target sound (default: "bird singing")',
    )
    parser.add_argument(
        "-m", "--model",
        choices=["small", "base", "large"],
        default="base",
        help="SAM-Audio model size (default: base)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device: cuda, cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Skip saving the residual (non-target) audio",
    )
    parser.add_argument(
        "--predict-spans",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable span prediction (default: enabled)",
    )
    parser.add_argument(
        "--reranking-candidates",
        type=int,
        default=1,
        help="Number of reranking candidates; higher = better quality, slower (default: 1)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not os.environ.get("HF_TOKEN"):
        logging.warning(
            "HF_TOKEN not set. Copy .env.sample to .env and add your token, "
            "or run: huggingface-cli login"
        )

    files = collect_audio_files(args.input)
    logging.info(
        "Found %d file(s) to process with prompt: \"%s\"",
        len(files),
        args.description,
    )

    separator = AudioSeparator(
        model_size=args.model,
        device=args.device,
    )

    for i, fpath in enumerate(files, 1):
        logging.info("[%d/%d] %s", i, len(files), fpath.name)
        try:
            separator.process_file(
                fpath,
                output_dir=args.output_dir,
                description=args.description,
                save_residual=not args.no_residual,
                predict_spans=args.predict_spans,
                reranking_candidates=args.reranking_candidates,
            )
        except Exception:
            logging.exception("Failed to process %s", fpath.name)

    logging.info("Done. Output saved to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()

# uv run python main.py /path/to/bird_audio.wav -d "bird singing" -m base -o ./output/ -v