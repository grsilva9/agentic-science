"""
Task: Use the configured LLM to write the narrative sections of the report.

This is the part that makes the pipeline genuinely "agentic" rather than
just LLM-dispatched: instead of stamping a fixed template into the PDF, we
ask the model to read the experiment's headline numbers and produce
appropriate, context-specific paragraphs.

The output is a dict keyed by section name. If LLM generation fails (model
unavailable, malformed JSON), we fall back to the static defaults from
``schemas._DEFAULT_REPORT_TEXT`` so the pipeline never breaks at the end.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict

import pandas as pd

from schemas import (
    WriteReportTextInput,
    WriteReportTextOutput,
    _DEFAULT_REPORT_TEXT,
)

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """\
You are writing the narrative sections of a short technical report for a
machine-learning experiment. The experiment trained a generative model on a
medical-imaging dataset and evaluated the synthetic samples with a
patch-based MMD permutation test.

Experiment details:
- Dataset: {dataset_name}
- Model: {model_description}
- Real samples used in test: {n_real_samples}
- Synthetic samples generated: {n_samples_generated}
- Observed MMD² statistic: {observed_mmd:.6g}
- Permutation-test p-value: {p_value:.4f}
- Decision at α = 0.05: {decision}

Null distribution summary (MMD² values from pooled-and-shuffled splits):
{null_summary}

Write five short paragraphs, one for each section listed below. Each paragraph
should be 2-4 sentences. Be precise about the statistics; do NOT invent any
numbers that are not in the experiment details. Avoid hype. Avoid bullet
points. Plain prose only.

Return ONLY a valid JSON object with these exact keys, nothing else:
- "introduction": Set up what was done and why MMD-based evaluation was used.
- "image_description": Describe what the synthetic image grid shows.
- "table_description": Describe the statistics table that follows.
- "plot_description": Describe what the permutation plot shows and how to read it.
- "conclusion": Interpret the p-value and the visual quality of samples honestly.

JSON:"""


def _decision_label(p_value: float, alpha: float = 0.05) -> str:
    if p_value < alpha:
        return f"reject H₀ (synthetic distribution differs from real)"
    return f"do not reject H₀ (synthetic distribution is statistically indistinguishable at α={alpha})"


def _build_prompt(input: WriteReportTextInput) -> str:
    df = pd.read_csv(str(input.stats_csv))
    # Format the null-summary lines for the prompt.
    summary_lines = []
    for row in df.itertuples(index=False):
        name = row[0]
        if not isinstance(name, str) or not name.startswith("null_"):
            continue
        try:
            value = float(row[1])
            summary_lines.append(f"  {name}: {value:.6g}")
        except (TypeError, ValueError):
            continue
    null_summary = "\n".join(summary_lines) if summary_lines else "  (unavailable)"

    return _PROMPT_TEMPLATE.format(
        dataset_name=input.dataset_name,
        model_description=input.model_description,
        n_real_samples=input.n_real_samples,
        n_samples_generated=input.n_samples_generated,
        observed_mmd=input.observed_mmd,
        p_value=input.p_value,
        decision=_decision_label(input.p_value),
        null_summary=null_summary,
    )


def _extract_json(text: str) -> Dict[str, str] | None:
    """
    Pull the first {...} block out of a text blob and parse it as JSON.

    Small models sometimes wrap JSON in prose or markdown code fences. We
    strip those before parsing.
    """
    # Drop markdown fences first.
    text = re.sub(r"```(?:json)?", "", text)

    # Find the first balanced-ish {...} block (greedy enough for a flat dict).
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned malformed JSON: %s", exc)
        return None


def write_report_text(input: WriteReportTextInput) -> WriteReportTextOutput:
    """
    Ask the configured LLM to write the report's narrative sections.

    Returns the LLM's text on success, or the static defaults on failure.
    The caller (a tool in agent.py) supplies the LLM by importing it from
    ``load_models``.
    """
    from load_models import call_llm_text  # local to avoid cycle on import

    prompt = _build_prompt(input)

    expected_keys = {
        "introduction",
        "image_description",
        "table_description",
        "plot_description",
        "conclusion",
    }
    fallback = dict(_DEFAULT_REPORT_TEXT)

    try:
        raw = call_llm_text(prompt)
    except Exception as exc:                       # noqa: BLE001
        logger.warning("LLM call failed (%s); using default report text.", exc)
        return WriteReportTextOutput(text_data=fallback)

    parsed = _extract_json(raw or "")
    if parsed is None or not expected_keys.issubset(parsed.keys()):
        missing = expected_keys - set(parsed.keys()) if parsed else expected_keys
        logger.warning(
            "LLM response missing keys %s; falling back to defaults.", missing
        )
        return WriteReportTextOutput(text_data=fallback)

    # Keep only the keys we asked for, drop anything weird.
    text_data = {k: str(parsed[k]).strip() for k in expected_keys}
    logger.info("LLM-generated report text accepted (lengths: %s)",
                {k: len(v) for k, v in text_data.items()})
    return WriteReportTextOutput(text_data=text_data)
