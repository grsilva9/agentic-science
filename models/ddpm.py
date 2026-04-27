"""
Denoising Diffusion Probabilistic Model (DDPM).

Implements the forward and reverse diffusion process with a cosine variance
schedule (Nichol & Dhariwal, 2021) and optional clipped reverse diffusion
for improved sample quality.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
import pytorch_lightning as pl

from models.unet import Unet


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------

class EMA(nn.Module):
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model whose weights are a running average
    of the training model's weights. Useful for more stable generation.

    See: https://github.com/Lightning-AI/pytorch-lightning/issues/10914
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(),
                model.state_dict().values(),
            ):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        """Update EMA weights toward the current model weights."""
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, model):
        """Hard-copy model weights into the EMA."""
        self._update(model, update_fn=lambda e, m: m)


# ---------------------------------------------------------------------------
# DDPM
# ---------------------------------------------------------------------------

class DDPM(pl.LightningModule):
    """
    DDPM with cosine variance schedule and ShuffleNet-based UNet.

    Args:
        image_size: Spatial dimension (assumes square images).
        in_channels: Number of image channels.
        time_embedding_dim: Dimension of the time embedding.
        timesteps: Number of diffusion steps.
        base_dim: Base channel dimension for the UNet.
        dim_mults: Channel multipliers per UNet stage.
        lr: Learning rate for AdamW.
        total_steps: Total optimizer steps for OneCycleLR scheduler.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        time_embedding_dim=256,
        timesteps=1000,
        base_dim=32,
        dim_mults=(1, 2, 4, 8),
        lr=5e-4,
        total_steps=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.timesteps = torch.tensor([timesteps], dtype=torch.long)
        self.in_channels = in_channels
        self.image_size = image_size
        self.lr = lr
        self.total_steps = total_steps

        # Noise schedule
        betas = self._cosine_variance_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.model = Unet(
            self.timesteps, time_embedding_dim, in_channels, in_channels,
            base_dim, dim_mults,
        )
        self.model_ema = EMA(self.model, decay=0.99)

    # --- Diffusion process ---

    def forward_diffusion(self, x_0, t, noise):
        """q(x_t | x_0): add noise to clean data at timestep t."""
        assert x_0.shape == noise.shape
        sqrt_alpha = self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def forward(self, x, noise):
        timesteps_int = int(self.timesteps.item())
        t = torch.randint(0, timesteps_int, (x.shape[0],), device=self.device)
        x_t = self.forward_diffusion(x, t, noise)
        return self.model(x_t, t)

    # --- Sampling ---

    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True):
        """
        Generate samples via iterative reverse diffusion.

        Returns samples in the model's native range [-1, 1]. Callers that
        want display-ready images (e.g. for matplotlib) should map them to
        [0, 1] via ``(samples + 1) / 2``.
        """
        x_t = torch.randn(
            (n_samples, self.in_channels, self.image_size, self.image_size),
            device=self.device,
        )
        reverse_fn = (
            self._reverse_diffusion_clipped if clipped_reverse_diffusion
            else self._reverse_diffusion
        )
        timesteps_int = int(self.timesteps.item())
        for i in tqdm(range(timesteps_int - 1, -1, -1), desc="Sampling"):
            noise = torch.randn_like(x_t)
            t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            x_t = reverse_fn(x_t, t, noise)
        return x_t.clamp(-1.0, 1.0)

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise):
        """p(x_{t-1} | x_t): standard reverse step."""
        pred = self.model(x_t, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_cp = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_omc = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)

        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1.0 - alpha_t) / sqrt_omc * pred)

        if t.min() > 0:
            alpha_cp_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
            std = torch.sqrt(beta_t * (1.0 - alpha_cp_prev) / (1.0 - alpha_cp))
        else:
            std = 0.0

        return mean + std * noise

    @torch.no_grad()
    def _reverse_diffusion_clipped(self, x_t, t, noise):
        """
        Clipped reverse diffusion: predict x_0, clip to [-1, 1], then
        compute the posterior mean. Improves sample quality.
        """
        pred = self.model(x_t, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_cp = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)

        x_0_pred = torch.sqrt(1.0 / alpha_cp) * x_t - torch.sqrt(1.0 / alpha_cp - 1.0) * pred
        x_0_pred.clamp_(-1.0, 1.0)

        if t.min() > 0:
            alpha_cp_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
            mean = (
                beta_t * torch.sqrt(alpha_cp_prev) / (1.0 - alpha_cp) * x_0_pred
                + (1.0 - alpha_cp_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_cp) * x_t
            )
            std = torch.sqrt(beta_t * (1.0 - alpha_cp_prev) / (1.0 - alpha_cp))
        else:
            mean = (beta_t / (1.0 - alpha_cp)) * x_0_pred
            std = 0.0

        return mean + std * noise

    # --- Noise schedule ---

    @staticmethod
    def _cosine_variance_schedule(timesteps, epsilon=0.008):
        """Cosine schedule from Nichol & Dhariwal (2021)."""
        steps = torch.linspace(0, timesteps.item(), steps=timesteps.item() + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    # --- Training ---

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, self.lr, total_steps=self.total_steps,
            pct_start=0.25, anneal_strategy="cos",
        )
        return [optimizer], [scheduler]

    def loss_function(self, pred_noise, true_noise):
        return nn.MSELoss()(pred_noise, true_noise)

    def training_step(self, batch, batch_idx):
        images = batch[0]
        noise = torch.randn_like(images)
        pred = self(images, noise)
        loss = self.loss_function(pred, noise)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return

    def on_before_backward(self, loss: Tensor) -> None:
        self.model_ema.update(self.model)

    def gen_sample(self, N=1):
        """Generate N samples (convenience wrapper)."""
        return self.sampling(n_samples=N)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_ddpm(
    timesteps=100,
    image_size=400,
    in_channel=1,
    base_dim=16,
    dim_mults=(2, 4),
    total_steps_factor=256,
):
    """Create a DDPM instance with the given hyperparameters."""
    total_steps = total_steps_factor * timesteps
    return DDPM(
        image_size=image_size,
        in_channels=in_channel,
        timesteps=timesteps,
        base_dim=base_dim,
        dim_mults=dim_mults,
        total_steps=total_steps,
    )
