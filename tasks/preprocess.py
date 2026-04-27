"""
Task: Preprocess raw images into train/test DataLoaders.

Two modes:

1. ``dataset_source="medmnist"`` (default): download a MedMNIST 2D subset
   (e.g. PneumoniaMNIST) and serialize its train/test split as DataLoaders.

2. ``dataset_source="png"``: load all .png files from ``raw_data_path``,
   resize them, then split by ``split_ratio``. This is the legacy mode
   from the original repo.

Pixel values can optionally be rescaled to [-1, 1], which is what the DDPM
implementation expects internally.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from schemas import PreprocessInput, PreprocessOutput
from tasks.dataset import load_medmnist

logger = logging.getLogger(__name__)


def _maybe_rescale(x: torch.Tensor, to_neg_one_one: bool) -> torch.Tensor:
    """Rescale [0, 1] tensors to [-1, 1] if requested."""
    return x * 2.0 - 1.0 if to_neg_one_one else x


def _load_pngs(raw_dir: Path, resize: tuple[int, int]) -> torch.Tensor:
    transform = T.ToTensor()
    tensors = []
    for img_fp in sorted(raw_dir.glob("*.png")):
        img = Image.open(img_fp).convert("L").resize(resize)
        tensors.append(transform(img))
    if not tensors:
        raise FileNotFoundError(f"No .png files found in {raw_dir}")
    return torch.stack(tensors)


def preprocess_task(input: PreprocessInput) -> PreprocessOutput:
    """
    Returns:
        PreprocessOutput with paths to (train_loader.pt, test_loader.pt).
    """
    if input.dataset_source == "medmnist":
        train_x, _, test_x, _, _ = load_medmnist(
            flag=input.medmnist_flag,
            size=input.medmnist_size,
        )
        train_data = _maybe_rescale(train_x, input.normalize_to_neg_one_one)
        test_data = _maybe_rescale(test_x, input.normalize_to_neg_one_one)
        logger.info(
            "MedMNIST '%s': %d train, %d test, shape %s",
            input.medmnist_flag, train_data.size(0), test_data.size(0),
            tuple(train_data.shape[1:]),
        )

    elif input.dataset_source == "png":
        if input.raw_data_path is None:
            raise ValueError("PNG mode requires `raw_data_path`.")
        data = _load_pngs(input.raw_data_path, input.resize)
        data = _maybe_rescale(data, input.normalize_to_neg_one_one)
        n = data.size(0)
        # Deterministic-ish shuffle for reproducibility.
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(n, generator=gen)
        split_idx = int(n * input.split_ratio)
        train_data = data[perm[:split_idx]]
        test_data = data[perm[split_idx:]]
        logger.info(
            "PNG mode: loaded %d images, %d train, %d test, shape %s",
            n, train_data.size(0), test_data.size(0), tuple(train_data.shape[1:]),
        )

    else:
        raise ValueError(f"Unknown dataset_source: {input.dataset_source}")

    train_loader = DataLoader(
        TensorDataset(train_data), batch_size=input.batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(test_data), batch_size=input.batch_size, shuffle=False,
    )

    out_dir = Path(input.preprocessed_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_loader.pt"
    test_path = out_dir / "test_loader.pt"
    torch.save(train_loader, str(train_path))
    torch.save(test_loader, str(test_path))
    logger.info("Saved DataLoaders to %s", out_dir)

    return PreprocessOutput(split_train_test=(train_path, test_path))
