"""
Agentic Science — LLM-powered ML research pipeline.

Usage:
    python agent.py prompt.txt
    python agent.py --prompt "Please run preprocess_data with dataset_source='medmnist'"

The agent dispatches one of seven tools:

    preprocess_data       — load + serialize a dataset to DataLoaders
    train_model           — train the DDPM
    generate_samples      — sample from a trained checkpoint
    evaluate_samples      — patch-MMD permutation test
    write_report_text     — LLM-narrated report sections (text only)
    create_report         — assemble the final PDF
    run_pipeline          — orchestrator: runs all of the above end-to-end
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from pydantic_ai import Agent, RunContext

from load_models import get_model
from schemas import (
    PreprocessInput,           PreprocessOutput,
    TrainInput,                TrainOutput,
    GenerateInput,             GenerateOutput,
    EvaluateInput,             EvaluateOutput,
    WriteReportTextInput,      WriteReportTextOutput,
    ReportInput,               ReportOutput,
    RunPipelineInput,          RunPipelineOutput,
)
from tasks.preprocess         import preprocess_task
from tasks.train              import train_model as train_model_task
from tasks.generate           import generate_task
from tasks.evaluate           import evaluate_task
from tasks.write_report_text  import write_report_text as write_report_text_task
from tasks.report             import report_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI research assistant. You have seven tools and must call exactly one
to satisfy each user request:

  1. preprocess_data    — load a dataset (default: PneumoniaMNIST from
                          MedMNIST) and write train/test DataLoaders to disk.
  2. train_model        — train a DDPM diffusion model on those DataLoaders.
  3. generate_samples   — sample synthetic images from a trained checkpoint.
  4. evaluate_samples   — run a patch-based MMD permutation test comparing
                          real test samples and generated samples.
  5. write_report_text  — ask the LLM (you, but in a separate call) to write
                          the narrative paragraphs of the final report.
  6. create_report      — assemble the PDF report from images, stats, and text.
  7. run_pipeline       — orchestrator that runs all of the above end-to-end.

Rules:

- Identify which single tool the user is asking for and call it.
- Fill in any unspecified arguments with the tool's documented defaults.
- If the user asks for the entire workflow ("run the full pipeline",
  "do everything"), call run_pipeline.
- Do not invent argument names that aren't in the tool signature.
- Do not chain multiple tool calls — each user prompt triggers ONE tool.

Examples:

User: "Run preprocess_data on PneumoniaMNIST."
→ call preprocess_data with dataset_source="medmnist", medmnist_flag="pneumoniamnist".

User: "Train for 15 epochs."
→ call train_model with max_epochs=15 and the default paths.

User: "Generate 32 samples from trained_models/model.ckpt."
→ call generate_samples with n_samples=32 and that checkpoint path.

User: "Evaluate against the test set with 1000 permutations."
→ call evaluate_samples with num_permutations=1000.

User: "Run the whole pipeline with PneumoniaMNIST and 5 epochs."
→ call run_pipeline with medmnist_flag="pneumoniamnist", max_epochs=5.
"""


agent = Agent(
    name="ImagePipelineAgent",
    model=get_model(),
    system_prompt=SYSTEM_PROMPT,
    deps_type=None,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@agent.tool
def preprocess_data(
    ctx: RunContext,
    dataset_source: str = "medmnist",
    medmnist_flag: str = "pneumoniamnist",
    medmnist_size: int = 28,
    raw_data_path: Optional[str] = None,
    normalize_to_neg_one_one: bool = True,
    resize: Tuple[int, int] = (28, 28),
    batch_size: int = 64,
    split_ratio: float = 0.8,
    preprocessed_path: str = "preprocessed_data/pneumoniamnist",
) -> Dict[str, Tuple[str, str]]:
    """
    Load a dataset and serialize train/test DataLoaders.

    Args:
        dataset_source: "medmnist" (default) or "png".
        medmnist_flag: MedMNIST 2D dataset name (only for "medmnist").
        medmnist_size: 28 / 64 / 128 / 224 (only for "medmnist").
        raw_data_path: directory of .png files (only for "png").
        normalize_to_neg_one_one: rescale [0,1] → [-1,1] for the DDPM.
        resize: target (H, W) for PNG mode.
        batch_size: DataLoader batch size.
        split_ratio: train fraction for PNG mode.
        preprocessed_path: where to save the .pt files.
    """
    logger.info("preprocess_data: source=%s flag=%s out=%s",
                dataset_source, medmnist_flag, preprocessed_path)
    inp = PreprocessInput(
        dataset_source=dataset_source,
        medmnist_flag=medmnist_flag,
        medmnist_size=medmnist_size,
        raw_data_path=Path(raw_data_path) if raw_data_path else None,
        normalize_to_neg_one_one=normalize_to_neg_one_one,
        resize=resize,
        batch_size=batch_size,
        split_ratio=split_ratio,
        preprocessed_path=Path(preprocessed_path),
    )
    out: PreprocessOutput = preprocess_task(inp)
    train_p, test_p = out.split_train_test
    return {"split_train_test": (str(train_p), str(test_p))}


@agent.tool
def train_model(
    ctx: RunContext,
    train_path: str = "preprocessed_data/pneumoniamnist/train_loader.pt",
    test_path: str = "preprocessed_data/pneumoniamnist/test_loader.pt",
    image_size: Tuple[int, int] = (28, 28),
    in_channel: int = 1,
    base_dim: int = 32,
    dim_mults: Tuple[int, int] = (2, 4),
    timesteps: int = 200,
    total_steps_factor: int = 50,
    max_epochs: int = 20,
    model_name: str = "model",
) -> Dict[str, str]:
    """Train a DDPM on serialized DataLoaders."""
    logger.info("train_model: epochs=%d model=%s", max_epochs, model_name)
    inp = TrainInput(
        split_train_test=(Path(train_path), Path(test_path)),
        image_size=image_size,
        in_channel=in_channel,
        base_dim=base_dim,
        dim_mults=list(dim_mults),
        timesteps=timesteps,
        total_steps_factor=total_steps_factor,
        max_epochs=max_epochs,
        model_name=model_name,
    )
    out: TrainOutput = train_model_task(inp)
    return {"model_checkpoint_path": str(out.model_checkpoint_path)}


@agent.tool
def generate_samples(
    ctx: RunContext,
    model_checkpoint_path: str = "trained_models/model.ckpt",
    n_samples: int = 64,
    samples_name: str = "samples",
) -> Dict[str, str]:
    """Generate synthetic images from a trained DDPM checkpoint."""
    ckpt = Path(model_checkpoint_path)
    if not ckpt.is_file():
        raise ValueError(f"Checkpoint not found: {model_checkpoint_path}")
    logger.info("generate_samples: ckpt=%s n=%d", model_checkpoint_path, n_samples)
    inp = GenerateInput(
        model_checkpoint_path=ckpt, n_samples=n_samples, samples_name=samples_name,
    )
    out: GenerateOutput = generate_task(inp)
    return {"gen_images_path": str(out.gen_images_path)}


@agent.tool
def evaluate_samples(
    ctx: RunContext,
    train_path: str = "preprocessed_data/pneumoniamnist/train_loader.pt",
    test_path: str = "preprocessed_data/pneumoniamnist/test_loader.pt",
    n_real_samples: int = 128,
    kernel_patch_size: int = 3,
    num_permutations: int = 1000,
    gen_images_path: str = "",
    evaluate_results_name: str = "Evaluation",
    seed: int = 0,
) -> Dict[str, Any]:
    """Patch-MMD permutation test of generated vs real samples."""
    logger.info("evaluate_samples: gen=%s permutations=%d",
                gen_images_path, num_permutations)
    inp = EvaluateInput(
        split_train_test=(Path(train_path), Path(test_path)),
        n_real_samples=n_real_samples,
        kernel_patch_size=kernel_patch_size,
        num_permutations=num_permutations,
        gen_images_path=Path(gen_images_path),
        evaluate_results_name=evaluate_results_name,
        seed=seed,
    )
    out: EvaluateOutput = evaluate_task(inp)
    return {
        "stats_csv": str(out.stats_csv),
        "plot_png":  str(out.plot_png),
        "p_value":   out.p_value,
        "observed_mmd": out.observed_mmd,
    }


@agent.tool
def write_report_text(
    ctx: RunContext,
    stats_csv: str,
    p_value: float,
    observed_mmd: float,
    n_samples_generated: int = 64,
    n_real_samples: int = 128,
    dataset_name: str = "PneumoniaMNIST",
    model_description: str = "DDPM with cosine variance schedule and ShuffleNet v2 UNet",
) -> Dict[str, Dict[str, str]]:
    """
    Have the LLM write the narrative sections of the final report based on
    the experiment's statistics. Falls back to defaults if generation fails.
    """
    logger.info("write_report_text: p_value=%.4f obs_mmd=%.4g", p_value, observed_mmd)
    inp = WriteReportTextInput(
        stats_csv=Path(stats_csv),
        p_value=p_value,
        observed_mmd=observed_mmd,
        n_samples_generated=n_samples_generated,
        n_real_samples=n_real_samples,
        dataset_name=dataset_name,
        model_description=model_description,
    )
    out: WriteReportTextOutput = write_report_text_task(inp)
    return {"text_data": out.text_data}


@agent.tool
def create_report(
    ctx: RunContext,
    gen_images_path: str,
    stats_csv: str,
    plot_png: str,
    text_data: Optional[Dict[str, Any]] = None,
    p_value: Optional[float] = None,
    observed_mmd: Optional[float] = None,
    title: str = "Agentic Science: Diffusion Model Report",
    report_name: str = "report",
) -> Dict[str, str]:
    """Assemble the PDF report."""
    logger.info("create_report: report_name=%s", report_name)
    kwargs = dict(
        gen_images_path=Path(gen_images_path),
        stats_csv=Path(stats_csv),
        plot_png=Path(plot_png),
        p_value=p_value,
        observed_mmd=observed_mmd,
        title=title,
        report_name=report_name,
    )
    if text_data is not None:
        kwargs["text_data"] = text_data
    out: ReportOutput = report_task(ReportInput(**kwargs))
    return {"report_path": str(out.report_path)}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@agent.tool
def run_pipeline(
    ctx: RunContext,
    dataset_source: str = "medmnist",
    medmnist_flag: str = "pneumoniamnist",
    raw_data_path: Optional[str] = None,
    image_size: int = 28,
    batch_size: int = 64,
    max_epochs: int = 10,
    n_samples: int = 64,
    n_real_samples: int = 128,
    num_permutations: int = 500,
    run_name: str = "run",
) -> Dict[str, Any]:
    """
    End-to-end pipeline: preprocess → train → generate → evaluate →
    write_report_text → create_report.

    A small local LLM cannot reliably chain six tool calls, so we provide a
    deterministic orchestrator. The model still picks the parameters; we
    just hard-code the order of operations.
    """
    cfg = RunPipelineInput(
        dataset_source=dataset_source,
        medmnist_flag=medmnist_flag,
        raw_data_path=Path(raw_data_path) if raw_data_path else None,
        image_size=image_size,
        batch_size=batch_size,
        max_epochs=max_epochs,
        n_samples=n_samples,
        n_real_samples=n_real_samples,
        num_permutations=num_permutations,
        run_name=run_name,
    )

    pre_out = preprocess_task(PreprocessInput(
        dataset_source=cfg.dataset_source,
        medmnist_flag=cfg.medmnist_flag,
        raw_data_path=cfg.raw_data_path,
        resize=(cfg.image_size, cfg.image_size),
        batch_size=cfg.batch_size,
        preprocessed_path=Path("preprocessed_data") / cfg.run_name,
    ))

    train_out = train_model_task(TrainInput(
        split_train_test=pre_out.split_train_test,
        image_size=(cfg.image_size, cfg.image_size),
        max_epochs=cfg.max_epochs,
        model_name=cfg.run_name,
    ))

    gen_out = generate_task(GenerateInput(
        model_checkpoint_path=train_out.model_checkpoint_path,
        n_samples=cfg.n_samples,
        samples_name=cfg.run_name,
    ))

    eval_out = evaluate_task(EvaluateInput(
        split_train_test=pre_out.split_train_test,
        n_real_samples=cfg.n_real_samples,
        num_permutations=cfg.num_permutations,
        gen_images_path=gen_out.gen_images_path,
        evaluate_results_name=cfg.run_name,
    ))

    text_out = write_report_text_task(WriteReportTextInput(
        stats_csv=eval_out.stats_csv,
        p_value=eval_out.p_value,
        observed_mmd=eval_out.observed_mmd,
        n_samples_generated=cfg.n_samples,
        n_real_samples=cfg.n_real_samples,
        dataset_name=cfg.medmnist_flag,
    ))

    report_out = report_task(ReportInput(
        gen_images_path=gen_out.gen_images_path,
        stats_csv=eval_out.stats_csv,
        plot_png=eval_out.plot_png,
        text_data=text_out.text_data,
        p_value=eval_out.p_value,
        observed_mmd=eval_out.observed_mmd,
        report_name=cfg.run_name,
        title=f"Agentic Science: {cfg.medmnist_flag} ({cfg.run_name})",
    ))

    result = RunPipelineOutput(
        preprocessed_path=pre_out.split_train_test,
        model_checkpoint_path=train_out.model_checkpoint_path,
        gen_images_path=gen_out.gen_images_path,
        stats_csv=eval_out.stats_csv,
        plot_png=eval_out.plot_png,
        report_path=report_out.report_path,
        p_value=eval_out.p_value,
        observed_mmd=eval_out.observed_mmd,
    )
    return {
        "preprocessed_path":     [str(p) for p in result.preprocessed_path],
        "model_checkpoint_path": str(result.model_checkpoint_path),
        "gen_images_path":       str(result.gen_images_path),
        "stats_csv":             str(result.stats_csv),
        "plot_png":              str(result.plot_png),
        "report_path":           str(result.report_path),
        "p_value":               result.p_value,
        "observed_mmd":          result.observed_mmd,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agentic Science — LLM research pipeline agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("prompt_file", nargs="?", help="Path to a .txt file containing the prompt.")
    group.add_argument("--prompt", "-p", type=str, help="Inline prompt string.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    prompt_string = args.prompt or Path(args.prompt_file).read_text()
    logger.info("Running agent with prompt: %s", prompt_string[:200])
    result = agent.run_sync(prompt_string, deps=None)

    print("---------------")
    print("Result:")
    print(result.output)


if __name__ == "__main__":
    main()
