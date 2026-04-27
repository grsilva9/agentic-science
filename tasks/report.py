"""
Task: Generate a structured PDF report.

Compiles generated images, evaluation statistics, and the permutation-test
plot into a single PDF using ReportLab. The narrative text can be supplied
externally (typically by the LLM via the ``write_report_text`` tool) or
fall back to sensible defaults from ``schemas._DEFAULT_REPORT_TEXT``.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from schemas import ReportInput, ReportOutput

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = Path("Reports")


def _to_display_range(images: torch.Tensor) -> torch.Tensor:
    """
    Map images to [0, 1] for visual display.

    The pipeline keeps generated samples in [-1, 1] (model space). For
    grayscale viewing we shift them. We also clip just in case the model
    produced slight overshoots.
    """
    if images.min() < -0.01:           # likely [-1, 1]
        return ((images + 1.0) / 2.0).clamp(0.0, 1.0)
    return images.clamp(0.0, 1.0)


def _render_image_grid(
    images_tensor: torch.Tensor, max_columns: int = 8
) -> str:
    """Render a grid of images to a temporary PNG. Returns its path."""
    images_tensor = _to_display_range(images_tensor.detach().cpu())
    images = images_tensor.numpy().transpose(0, 2, 3, 1)        # [N, H, W, C]
    num = len(images)
    cols = min(max_columns, num)
    rows = math.ceil(num / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.atleast_2d(axes).flatten() if num > 1 else np.array([axes])

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < num:
            img = images[i]
            if img.ndim == 3 and img.shape[2] == 1:
                img = img.squeeze(-1)
            ax.imshow(
                np.clip(img * 255, 0, 255).astype(np.uint8),
                cmap="gray" if img.ndim == 2 else None,
            )

    fig.tight_layout(pad=0.5)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def report_task(input: ReportInput) -> ReportOutput:
    """
    Build a structured PDF.

    Sections:
        1. Title and introduction.
        2. Generated image grid.
        3. Headline statistics (p-value, observed MMD).
        4. Full statistics table.
        5. Permutation-test plot.
        6. Conclusion.
    """
    output_dir = _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{input.report_name}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54,
    )
    styles = getSampleStyleSheet()
    story = []

    # ---- Title ----
    story.append(Paragraph(f"<b>{input.title}</b>", styles["Title"]))
    story.append(Spacer(1, 18))

    # ---- Introduction ----
    story.append(Paragraph("<b>Introduction</b>", styles["Heading2"]))
    story.append(Paragraph(input.text_data.get("introduction", ""), styles["BodyText"]))
    story.append(Spacer(1, 14))

    # ---- Generated images ----
    story.append(Paragraph("<b>Generated samples</b>", styles["Heading2"]))
    story.append(Paragraph(
        input.text_data.get("image_description", ""), styles["BodyText"]
    ))
    story.append(Spacer(1, 8))

    images_tensor = torch.load(str(input.gen_images_path), weights_only=False)
    grid_path = _render_image_grid(images_tensor)
    story.append(RLImage(grid_path, width=460, height=240))
    story.append(Spacer(1, 14))

    # ---- Headline result box ----
    if input.p_value is not None and input.observed_mmd is not None:
        story.append(Paragraph("<b>Headline result</b>", styles["Heading2"]))
        result_data = [
            ["Metric", "Value"],
            ["Observed MMD²", f"{input.observed_mmd:.6g}"],
            ["p-value", f"{input.p_value:.4f}"],
            ["Decision (α = 0.05)",
             "reject H₀" if input.p_value < 0.05 else "do not reject H₀"],
        ]
        headline = Table(result_data, colWidths=[200, 200])
        headline.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b78c2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(headline)
        story.append(Spacer(1, 14))

    # ---- Stats table ----
    story.append(Paragraph("<b>Permutation test statistics</b>", styles["Heading2"]))
    story.append(Paragraph(
        input.text_data.get("table_description", ""), styles["BodyText"]
    ))
    story.append(Spacer(1, 8))

    df = pd.read_csv(str(input.stats_csv))
    # Limit numeric precision for readability.
    formatted = []
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        formatted.append(cells)
    table_data = [list(df.columns)] + formatted
    table = Table(table_data, colWidths=[160, 120])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    # ---- Plot ----
    story.append(Paragraph("<b>Permutation test plot</b>", styles["Heading2"]))
    story.append(Paragraph(
        input.text_data.get("plot_description", ""), styles["BodyText"]
    ))
    story.append(Spacer(1, 8))
    story.append(RLImage(str(input.plot_png), width=440, height=260))
    story.append(Spacer(1, 14))

    # ---- Conclusion ----
    if input.text_data.get("conclusion"):
        story.append(Paragraph("<b>Conclusion</b>", styles["Heading2"]))
        story.append(Paragraph(input.text_data["conclusion"], styles["BodyText"]))

    doc.build(story)
    logger.info("Report saved to %s", pdf_path)

    Path(grid_path).unlink(missing_ok=True)
    return ReportOutput(report_path=pdf_path)
