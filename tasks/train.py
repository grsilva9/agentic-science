"""
Task: Train a DDPM diffusion model.

Loads serialized DataLoaders, builds a DDPM with the specified architecture,
trains with PyTorch Lightning, and saves the checkpoint.

Validation is disabled during training: for an unconditional DDPM, the
validation loss is just the noise-prediction MSE on the test set, which is
not a useful early-stopping signal. The real evaluation runs separately
via the patch-MMD permutation test.
"""

from __future__ import annotations

from pytorch_lightning.callbacks import TQDMProgressBar
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl

from models.ddpm import create_ddpm
from schemas import TrainInput, TrainOutput

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path("trained_models")


def train_model(input: TrainInput) -> TrainOutput:
    """
    Pipeline:
        1. Load the train DataLoader from disk.
        2. Build a DDPM with the given hyperparameters.
        3. Train with PyTorch Lightning.
        4. Save the checkpoint.
    """
    train_loader_path, _ = input.split_train_test

    logger.info("Loading train DataLoader: %s", train_loader_path)
    train_loader = torch.load(str(train_loader_path), weights_only=False)

    model = create_ddpm(
        timesteps=input.timesteps,
        image_size=input.image_size[0],
        in_channel=input.in_channel,
        base_dim=input.base_dim,
        dim_mults=list(input.dim_mults),
        total_steps_factor=input.total_steps_factor,
    )
    logger.info(
        "DDPM created: timesteps=%d, image_size=%d, base_dim=%d, dim_mults=%s",
        input.timesteps, input.image_size[0], input.base_dim, input.dim_mults,
    )

    trainer = pl.Trainer(
        max_epochs=input.max_epochs,
        log_every_n_steps=10,
        enable_checkpointing=False,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        num_sanity_val_steps=0,        # skip the pre-flight sanity check
        limit_val_batches=0,           # disable validation during training
        enable_model_summary=False,    # skip the parameter-count table
    )
    trainer.fit(model=model, train_dataloaders=train_loader)

    model_dir = _DEFAULT_MODEL_DIR
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / f"{input.model_name}.ckpt"
    trainer.save_checkpoint(str(model_path))
    logger.info("Checkpoint saved to %s", model_path)

    return TrainOutput(model_checkpoint_path=model_path)