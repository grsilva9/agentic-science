"""
Defines all schemas related to input and output of each process.
"""

from pydantic import BaseModel
from pathlib import Path
from typing import Tuple, Optional, Dict

#1. Defines input and output.

class PreprocessInput(BaseModel):
    raw_data_path: Path
    normalize: bool = True
    resize: Tuple[int, int] = (64, 64)
    batch_size: int = 4
    split_ratio: float = 0.8
    preprocessed_path: Path = Path("preprocessed_data/noise_images")

class PreprocessOutput(BaseModel):
    split_train_test: Tuple[Path, Path]

class TrainInput(BaseModel):
    split_train_test: Tuple[Path, Path]
    image_size: Tuple[int, int] = (400, 400)
    in_channel: int = 1
    base_dim: int = 16
    dim_mults: Tuple[int, int] = [2, 4]
    timesteps: int = 100
    total_steps_factor: int = 256
    max_epochs: int = 100_000
    model_name: str = 'model'

class TrainOutput(BaseModel):
    model_checkpoint_path: Path

class GenerateInput(BaseModel):
    model_checkpoint_path: Path
    n_samples: int = 16
    samples_name: str = 'sample'

class GenerateOutput(BaseModel):
    gen_images_path: Path

class EvaluateInput(BaseModel):
    split_train_test: Tuple[Path, Path]
    n_test_data: int = 1000
    bootstrap_data: bool = True
    kernel_patch_size: int = 3
    num_permutations: int = 1000
    gen_images_path: Path
    evaluate_results_name: 'str' = 'Evaluation'

class EvaluateOutput(BaseModel):
    stats_csv: Path
    plot_png: Path

class ReportInput(BaseModel):
    gen_images_path: Path
    stats_csv: Path
    plot_png: Path
    text_data: Dict
    report_name: str = 'report'

class ReportOutput(BaseModel):
    report_path: Path

