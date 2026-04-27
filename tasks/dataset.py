"""
Dataset helpers.

Provides a single entry point ``load_medmnist`` that downloads (if needed)
and returns a MedMNIST 2D dataset as PyTorch tensors.

We keep this in its own module so the rest of the pipeline doesn't need to
know whether the data came from MedMNIST, a folder of PNGs, or anything
else.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Map from MedMNIST flag → number of channels. Most are grayscale; a few are RGB.
_RGB_FLAGS = {"pathmnist", "dermamnist", "bloodmnist", "retinamnist"}


def _flag_channels(flag: str) -> int:
    return 3 if flag.lower() in _RGB_FLAGS else 1


def load_medmnist(
    flag: str = "pneumoniamnist",
    size: int = 28,
    cache_dir: Path | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Download (if needed) and load a MedMNIST 2D dataset.

    Args:
        flag: MedMNIST dataset name, e.g. ``"pneumoniamnist"``.
        size: Resolution. Must be one of {28, 64, 128, 224}. The ``medmnist``
              package will refuse other values.
        cache_dir: Where to cache the .npz file. Defaults to ``~/.medmnist``.

    Returns:
        ``(train_x, train_y, test_x, test_y, channels)`` where ``train_x``
        and ``test_x`` are float tensors in [0, 1] of shape [N, C, H, W].
        Labels are int64 tensors of shape [N] (squeezed). ``channels`` is
        1 (grayscale) or 3 (RGB) depending on the dataset.
    """
    try:
        import medmnist  # noqa: F401
        from medmnist import INFO
        from medmnist.dataset import MedMNIST2D
    except ImportError as exc:
        raise ImportError(
            "medmnist is not installed. Run `pip install medmnist`."
        ) from exc

    if flag not in INFO:
        valid = ", ".join(sorted(k for k in INFO if not k.endswith("3d")))
        raise ValueError(f"Unknown MedMNIST flag '{flag}'. Valid 2D flags: {valid}")

    channels = _flag_channels(flag)

    # Build the dataset class dynamically. medmnist's per-class objects are
    # the canonical way to access each subset.
    dataset_cls_name = INFO[flag]["python_class"]  # e.g. "PneumoniaMNIST"
    dataset_cls = getattr(__import__("medmnist", fromlist=[dataset_cls_name]),
                          dataset_cls_name)

    kwargs = {"download": True}
    if cache_dir is not None:
        kwargs["root"] = str(Path(cache_dir).expanduser())
    if size != 28:
        kwargs["size"] = size  # MedMNIST+ resolution

    train_set = dataset_cls(split="train", **kwargs)
    test_set = dataset_cls(split="test", **kwargs)

    def _to_tensor(ds) -> Tuple[torch.Tensor, torch.Tensor]:
        # medmnist exposes `.imgs` (uint8) and `.labels` arrays.
        imgs = np.asarray(ds.imgs)
        labels = np.asarray(ds.labels).squeeze().astype(np.int64)

        # Handle grayscale-vs-RGB. MedMNIST stores grayscale as [N, H, W].
        if imgs.ndim == 3:
            imgs = imgs[:, None, :, :]                # -> [N, 1, H, W]
        elif imgs.ndim == 4:
            imgs = imgs.transpose(0, 3, 1, 2)         # -> [N, C, H, W]
        else:
            raise ValueError(f"Unexpected image array shape: {imgs.shape}")

        x = torch.from_numpy(imgs).float() / 255.0    # [0, 1]
        y = torch.from_numpy(labels)
        return x, y

    train_x, train_y = _to_tensor(train_set)
    test_x, test_y = _to_tensor(test_set)

    logger.info(
        "Loaded MedMNIST '%s' (size=%d): train=%s, test=%s, channels=%d",
        flag, size, tuple(train_x.shape), tuple(test_x.shape), channels,
    )
    return train_x, train_y, test_x, test_y, channels
