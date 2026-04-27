"""
Task: Generate synthetic images from a trained DDPM checkpoint.

The samples are saved in the same value range as the training data
(typically [-1, 1] when the preprocess task ran with
``normalize_to_neg_one_one=True``). The report task rescales them for
display.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from models.ddpm import DDPM
from schemas import GenerateInput, GenerateOutput

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = Path("generated_samples")


def generate_task(input: GenerateInput) -> GenerateOutput:
    ckpt = Path(input.model_checkpoint_path)
    logger.info("Loading checkpoint: %s", ckpt)
    model = DDPM.load_from_checkpoint(str(ckpt))
    model.eval()

    logger.info("Generating %d samples...", input.n_samples)
    samples = model.gen_sample(N=input.n_samples)            # in [-1, 1]
    samples = samples.detach().cpu()

    output_dir = _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input.samples_name}.pt"
    torch.save(samples, str(output_path))
    logger.info(
        "Saved %d samples to %s (range: [%.3f, %.3f])",
        samples.size(0), output_path, samples.min().item(), samples.max().item(),
    )

    return GenerateOutput(gen_images_path=output_path)
