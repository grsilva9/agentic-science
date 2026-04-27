"""
Pydantic schemas for pipeline input/output validation.

Each task (preprocess, train, generate, evaluate, write_report_text, report,
run_pipeline) has a corresponding Input and Output model. The agent tools
convert raw arguments into these models before calling the task functions.
"""

from pydantic import BaseModel, Field
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Literal


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------

class PreprocessInput(BaseModel):
    """
    Inputs to ``preprocess_task``.

    The pipeline supports two data sources:

    * ``"medmnist"``  — Download a MedMNIST 2D dataset (default:
      pneumoniamnist) and serialize it as train/test DataLoaders. No
      ``raw_data_path`` is required.
    * ``"png"``       — Load all .png files from ``raw_data_path``. This is
      the original behavior, kept for backward compatibility.
    """
    dataset_source: Literal["medmnist", "png"] = Field(
        "medmnist", description="Where the raw data comes from."
    )
    medmnist_flag: str = Field(
        "pneumoniamnist",
        description="MedMNIST dataset flag (e.g. 'pneumoniamnist', 'bloodmnist').",
    )
    medmnist_size: int = Field(
        28, description="MedMNIST resolution: 28, 64, 128, or 224."
    )
    raw_data_path: Optional[Path] = Field(
        None, description="Directory containing raw .png images (PNG mode only)."
    )
    normalize_to_neg_one_one: bool = Field(
        True,
        description=(
            "If True, scale pixels from [0, 1] to [-1, 1]. The DDPM expects "
            "[-1, 1] inputs."
        ),
    )
    resize: Tuple[int, int] = Field(
        (28, 28), description="Target (height, width) for resizing."
    )
    batch_size: int = Field(64, description="Batch size for the DataLoaders.")
    split_ratio: float = Field(
        0.8,
        description=(
            "Fraction of data used for training (only relevant for PNG mode; "
            "MedMNIST has its own train/test split)."
        ),
    )
    preprocessed_path: Path = Field(
        Path("preprocessed_data/pneumoniamnist"),
        description="Output directory for saved DataLoaders.",
    )


class PreprocessOutput(BaseModel):
    split_train_test: Tuple[Path, Path] = Field(
        ..., description="Paths to (train_loader.pt, test_loader.pt)."
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

class TrainInput(BaseModel):
    split_train_test: Tuple[Path, Path] = Field(
        ..., description="Paths to (train_loader.pt, test_loader.pt)."
    )
    image_size: Tuple[int, int] = Field((28, 28), description="Spatial dimensions (H, W).")
    in_channel: int = Field(1, description="Number of image channels (1 for grayscale).")
    base_dim: int = Field(32, description="Base channel dimension for the UNet.")
    dim_mults: List[int] = Field([2, 4], description="Channel multipliers per UNet stage.")
    timesteps: int = Field(200, description="Number of diffusion timesteps.")
    total_steps_factor: int = Field(
        50, description="Multiplier for total optimizer steps (timesteps * factor)."
    )
    max_epochs: int = Field(20, description="Maximum training epochs.")
    model_name: str = Field("model", description="Filename stem for the saved checkpoint.")


class TrainOutput(BaseModel):
    model_checkpoint_path: Path


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

class GenerateInput(BaseModel):
    model_checkpoint_path: Path = Field(..., description="Path to a trained .ckpt file.")
    n_samples: int = Field(64, description="Number of images to generate.")
    samples_name: str = Field("samples", description="Filename stem for the saved tensor.")


class GenerateOutput(BaseModel):
    gen_images_path: Path


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

class EvaluateInput(BaseModel):
    split_train_test: Tuple[Path, Path] = Field(
        ..., description="Paths to (train_loader.pt, test_loader.pt)."
    )
    n_real_samples: int = Field(
        128,
        description=(
            "Number of real samples to draw from the test loader for the "
            "two-sample test."
        ),
    )
    kernel_patch_size: int = Field(3, description="Patch size for the MMD kernel.")
    num_permutations: int = Field(1000, description="Number of permutation iterations.")
    gen_images_path: Path = Field(..., description="Path to generated samples .pt file.")
    evaluate_results_name: str = Field("Evaluation", description="Subfolder name for results.")
    seed: int = Field(0, description="RNG seed for reproducibility.")


class EvaluateOutput(BaseModel):
    stats_csv: Path
    plot_png: Path
    p_value: float
    observed_mmd: float


# ---------------------------------------------------------------------------
# Write report text (LLM-generated narrative)
# ---------------------------------------------------------------------------

class WriteReportTextInput(BaseModel):
    """Inputs for the LLM-narrated report text."""
    stats_csv: Path = Field(..., description="Path to evaluation stats CSV.")
    p_value: float = Field(..., description="Permutation test p-value.")
    observed_mmd: float = Field(..., description="Observed MMD statistic.")
    n_samples_generated: int = Field(..., description="How many synthetic images were generated.")
    n_real_samples: int = Field(..., description="How many real images were used in the test.")
    dataset_name: str = Field("PneumoniaMNIST", description="Human-readable dataset name.")
    model_description: str = Field(
        "DDPM with cosine variance schedule and ShuffleNet v2 UNet",
        description="Short technical description of the generative model.",
    )


class WriteReportTextOutput(BaseModel):
    text_data: Dict[str, str] = Field(
        ...,
        description=(
            "Dict with keys 'introduction', 'image_description', "
            "'table_description', 'plot_description', 'conclusion'."
        ),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_DEFAULT_REPORT_TEXT: Dict[str, str] = {
    "introduction": (
        "This automated report summarizes the results of a small-scale "
        "diffusion-model experiment. We train a DDPM on a medical imaging "
        "dataset, generate synthetic samples, and quantify how well the "
        "generated distribution matches the real one using a patch-based "
        "MMD permutation test."
    ),
    "image_description": (
        "The grid below shows a random sample of synthetic images produced "
        "by the trained DDPM after reverse diffusion."
    ),
    "table_description": (
        "Summary statistics of the null and observed MMD distributions "
        "produced by the permutation test."
    ),
    "plot_description": (
        "Density of the null distribution (blue) and the location of the "
        "observed MMD statistic (red dashed line). When the observed value "
        "lies inside the bulk of the null, we cannot reject the hypothesis "
        "that real and generated samples come from the same distribution."
    ),
    "conclusion": (
        "Interpret the p-value alongside the visual quality of generated "
        "samples — a high p-value combined with reasonable-looking samples "
        "is encouraging; a low p-value indicates the model has not yet "
        "captured the data distribution."
    ),
}


class ReportInput(BaseModel):
    gen_images_path: Path = Field(..., description="Path to generated samples .pt file.")
    stats_csv: Path = Field(..., description="Path to evaluation statistics CSV.")
    plot_png: Path = Field(..., description="Path to permutation test plot PNG.")
    text_data: Dict[str, Any] = Field(
        default_factory=lambda: dict(_DEFAULT_REPORT_TEXT),
        description="Section text for the report.",
    )
    p_value: Optional[float] = Field(None, description="Permutation test p-value (optional).")
    observed_mmd: Optional[float] = Field(None, description="Observed MMD (optional).")
    report_name: str = Field("report", description="Filename stem for the output PDF.")
    title: str = Field(
        "Agentic Science: Diffusion Model Report",
        description="Title shown at the top of the PDF.",
    )


class ReportOutput(BaseModel):
    report_path: Path


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class RunPipelineInput(BaseModel):
    """
    End-to-end pipeline configuration.

    The orchestrator runs preprocess → train → generate → evaluate →
    write_report_text → report deterministically. The LLM still chooses
    the parameters but doesn't have to chain tool calls.
    """
    dataset_source: Literal["medmnist", "png"] = "medmnist"
    medmnist_flag: str = "pneumoniamnist"
    raw_data_path: Optional[Path] = None
    image_size: int = 28
    batch_size: int = 64
    max_epochs: int = 10
    n_samples: int = 64
    n_real_samples: int = 128
    num_permutations: int = 500
    run_name: str = "run"


class RunPipelineOutput(BaseModel):
    preprocessed_path: Tuple[Path, Path]
    model_checkpoint_path: Path
    gen_images_path: Path
    stats_csv: Path
    plot_png: Path
    report_path: Path
    p_value: float
    observed_mmd: float
