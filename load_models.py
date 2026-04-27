"""
LLM backend configuration.

Set the AGENT_MODEL environment variable to choose a backend:

    AGENT_MODEL=ollama          (default) Local Llama 3.2 via Ollama
    AGENT_MODEL=ollama-qwen     Local Qwen 2.5 via Ollama (better tool-use)
    AGENT_MODEL=groq            Groq-hosted model        (GROQ_API_KEY)
    AGENT_MODEL=gemini          Google Gemini            (GOOGLE_API_KEY)
    AGENT_MODEL=anthropic       Anthropic Claude         (ANTHROPIC_API_KEY)

Two entry points:
    get_model()       -> a pydantic-ai Model used by the dispatcher Agent.
    call_llm_text(p)  -> a plain text-in / text-out helper used by tasks
                          that need free-form generation (e.g. report text).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_BACKEND = os.environ.get("AGENT_MODEL", "ollama").lower()

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Default model names per backend. All overridable via *_MODEL_NAME env vars.
_DEFAULTS = {
    "ollama":      ("llama3.2",                 "OLLAMA_MODEL_NAME"),
    "ollama-qwen": ("qwen2.5:3b",               "OLLAMA_MODEL_NAME"),
    "groq":        ("llama-3.3-70b-versatile",  "GROQ_MODEL_NAME"),
    "gemini":      ("gemini-2.0-flash",         "GEMINI_MODEL_NAME"),
    "anthropic":   ("claude-3-5-sonnet-latest", "ANTHROPIC_MODEL_NAME"),
}


def _resolve_model_name(backend: str) -> str:
    default, env_var = _DEFAULTS.get(backend, ("", "OLLAMA_MODEL_NAME"))
    return os.environ.get(env_var, default)


# ---------------------------------------------------------------------------
# pydantic-ai Model factory
# ---------------------------------------------------------------------------

def _build_ollama_model():
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
    name = _resolve_model_name(_BACKEND)
    logger.info("Using Ollama backend: model=%s, url=%s", name, _OLLAMA_BASE_URL)
    return OpenAIModel(
        model_name=name,
        provider=OpenAIProvider(base_url=_OLLAMA_BASE_URL),
    )


def _build_groq_model():
    from pydantic_ai.models.groq import GroqModel
    name = _resolve_model_name("groq")
    logger.info("Using Groq backend: model=%s", name)
    return GroqModel(name)


def _build_gemini_model():
    from pydantic_ai.models.gemini import GeminiModel
    name = _resolve_model_name("gemini")
    logger.info("Using Gemini backend: model=%s", name)
    return GeminiModel(name)


def _build_anthropic_model():
    from pydantic_ai.models.anthropic import AnthropicModel
    name = _resolve_model_name("anthropic")
    logger.info("Using Anthropic backend: model=%s", name)
    return AnthropicModel(name)


_BUILDERS = {
    "ollama":      _build_ollama_model,
    "ollama-qwen": _build_ollama_model,
    "groq":        _build_groq_model,
    "gemini":      _build_gemini_model,
    "anthropic":   _build_anthropic_model,
}


def get_model():
    """Return the pydantic-ai Model used by the dispatcher Agent."""
    builder = _BUILDERS.get(_BACKEND)
    if builder is None:
        raise ValueError(
            f"Unknown AGENT_MODEL='{_BACKEND}'. "
            f"Valid options: {', '.join(_BUILDERS)}"
        )
    return builder()


# ---------------------------------------------------------------------------
# Plain text-in / text-out helper
# ---------------------------------------------------------------------------

def call_llm_text(prompt: str, max_tokens: int = 1024) -> str:
    """
    Send a prompt to the configured LLM and return its raw text output.

    This is used for free-form generation (e.g. writing report sections).
    For Ollama backends we hit the OpenAI-compatible /v1/chat/completions
    endpoint directly; for hosted backends we route through their official
    SDKs.
    """
    if _BACKEND.startswith("ollama"):
        return _call_ollama_text(prompt, max_tokens)
    if _BACKEND == "groq":
        return _call_groq_text(prompt, max_tokens)
    if _BACKEND == "gemini":
        return _call_gemini_text(prompt, max_tokens)
    if _BACKEND == "anthropic":
        return _call_anthropic_text(prompt, max_tokens)
    raise ValueError(f"Unsupported backend for text generation: {_BACKEND}")


def _call_ollama_text(prompt: str, max_tokens: int) -> str:
    import httpx
    name = _resolve_model_name(_BACKEND)
    payload = {
        "model": name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": max_tokens,
        "stream": False,
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{_OLLAMA_BASE_URL}/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _call_groq_text(prompt: str, max_tokens: int) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    name = _resolve_model_name("groq")
    resp = client.chat.completions.create(
        model=name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""


def _call_gemini_text(prompt: str, max_tokens: int) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(_resolve_model_name("gemini"))
    resp = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": 0.4},
    )
    return getattr(resp, "text", "") or ""


def _call_anthropic_text(prompt: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    name = _resolve_model_name("anthropic")
    resp = client.messages.create(
        model=name,
        max_tokens=max_tokens,
        temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )
    # Concatenate text blocks.
    return "".join(b.text for b in resp.content if getattr(b, "text", None))
