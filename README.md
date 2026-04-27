# Agentic Science

An LLM-powered agent that automates the end-to-end workflow of a machine-learning researcher — from data preprocessing to a PDF report whose narrative paragraphs are written by the LLM itself — using natural-language commands.

## Why this exists

Machine-learning research has a repetitive shape: clean the data, train a model, generate outputs, evaluate them statistically, write up the results. Each step has its own scripts, arguments, and file paths. **Agentic Science** wraps the whole loop behind a single agent that understands plain English and dispatches the right tool with the right parameters.

The default backend (Llama 3.2 via Ollama) runs entirely on-device — no API keys, no cloud — but the same code can be pointed at Groq, Gemini, or Anthropic with a single environment variable.

## What it does

The agent has seven tools:

| Step | Tool | What it does |
|------|------|-------------|
| 1 | `preprocess_data` | Load a dataset (default: PneumoniaMNIST from MedMNIST), serialize as DataLoaders. |
| 2 | `train_model` | Train a Denoising Diffusion Probabilistic Model with a ShuffleNet-v2-based UNet. |
| 3 | `generate_samples` | Sample synthetic images via reverse diffusion. |
| 4 | `evaluate_samples` | Patch-based MMD permutation test — proper p-value for "real == generated". |
| 5 | `write_report_text` | The LLM reads the experiment's stats and writes the report's narrative sections. |
| 6 | `create_report` | Assemble the PDF (title, image grid, headline result, stats table, plot, conclusion). |
| 7 | `run_pipeline` | Orchestrator: runs all of the above end-to-end. |

A typical prompt looks like:

```
Please run preprocess_data with dataset_source='medmnist',
medmnist_flag='pneumoniamnist', batch_size=64.
```

The agent parses the intent, maps it to the correct tool, fills in defaults for anything unspecified, and executes it.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                User prompt (natural language)              │
└──────────────────────────┬────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────┐
│                PydanticAI Agent (agent.py)                 │
│                                                           │
│   LLM:  Llama 3.2 / Qwen 2.5 (Ollama, local)              │
│         or Groq / Gemini / Anthropic (hosted)              │
│   Typed tool dispatch with Pydantic schemas                │
└──┬───────┬───────┬───────┬─────────┬─────────┬────────────┘
   │       │       │       │         │         │
   ▼       ▼       ▼       ▼         ▼         ▼
┌───────┬───────┬────────┬────────┬─────────┬────────┐
│Preproc│ Train │Generate│Evaluate│  Write  │ Report │
│       │ (DDPM)│        │ (Patch │  Report │  PDF   │
│       │       │        │  MMD)  │  Text   │        │
└───────┴───────┴────────┴────────┴─────────┴────────┘
   │       │       │       │         │         │
   ▼       ▼       ▼       ▼         ▼         ▼
   .pt   .ckpt    .pt    CSV/PNG   text JSON   PDF
                                              run_pipeline (orchestrator)
                                              chains all of the above
```

### Key technical choices

- **PydanticAI** for agent–tool binding with typed input/output schemas; every tool call is validated before execution.
- **DDPM with a cosine variance schedule** and a ShuffleNet-v2-based UNet — channel-split, depthwise convolution, and channel-shuffle keep the parameter count low while remaining expressive.
- **Patch-based MMD** with a Gaussian kernel for evaluation — a kernel two-sample test on local image patches rather than on global features. Bandwidth via the median heuristic on real data; p-value via a one-sided permutation test with the Phipson–Smyth `(1 + #{null ≥ obs}) / (1 + N)` correction.
- **LLM-narrated report.** The model reads the actual stats and writes context-specific paragraphs; if generation fails, the pipeline falls back to static defaults.
- **Deterministic orchestrator.** Small local LLMs are unreliable at chaining six tool calls; `run_pipeline` does the chaining in Python while the LLM still chooses the parameters.
- **Pluggable LLM backend.** `AGENT_MODEL=ollama|ollama-qwen|groq|gemini|anthropic` with the appropriate API key.

## Repository layout

```
agentic-science/
├── agent.py                    # Agent + tool definitions + CLI
├── load_models.py              # Backend selection (Ollama / Groq / Gemini / Anthropic)
├── schemas.py                  # Pydantic Input/Output models for every task
├── requirements.txt
├── models/
│   ├── ddpm.py                 # DDPM (PyTorch Lightning)
│   └── unet.py                 # ShuffleNet-v2 UNet
├── tasks/
│   ├── dataset.py              # MedMNIST loader
│   ├── preprocess.py           # MedMNIST or PNG → DataLoaders
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py             # Patch-MMD permutation test
│   ├── write_report_text.py    # LLM-narrated report sections
│   └── report.py               # PDF assembly
├── tests/
│   └── test_permutation.py     # Sanity checks for the permutation logic
└── agentic_scientist.ipynb     # End-to-end Colab demo
```

## Getting started

### Local

```bash
git clone https://github.com/grsilva9/agentic-science.git
cd agentic-science

pip install -r requirements.txt

# Local LLM via Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2

# Run a single tool…
python agent.py --prompt "Please run preprocess_data with dataset_source='medmnist'."

# …or run the entire pipeline
python agent.py --prompt "Please run run_pipeline with medmnist_flag='pneumoniamnist', max_epochs=5, run_name='demo'."
```

### Google Colab

Open `agentic_scientist.ipynb` in Colab. The notebook installs dependencies, spins up Ollama in-runtime, pulls Llama 3.2, runs the unit tests for the permutation logic, walks through each step, and finishes with the orchestrator demo.

## Switching LLM backends

| Backend       | env var           | API key env var       | Notes                              |
|---------------|-------------------|-----------------------|------------------------------------|
| `ollama`      | (default)         | —                     | Llama 3.2 3B locally.              |
| `ollama-qwen` | `AGENT_MODEL`     | —                     | Qwen 2.5 3B locally.               |
| `groq`        | `AGENT_MODEL`     | `GROQ_API_KEY`        | Hosted Llama 3.3 70B.              |
| `gemini`      | `AGENT_MODEL`     | `GOOGLE_API_KEY`      | Gemini 2.0 Flash.                  |
| `anthropic`   | `AGENT_MODEL`     | `ANTHROPIC_API_KEY`   | Claude.                            |

```bash
AGENT_MODEL=groq GROQ_API_KEY=... python agent.py --prompt "..."
```

## Example prompts

**Preprocess:**
```
Please run preprocess_data with dataset_source='medmnist',
medmnist_flag='pneumoniamnist', batch_size=64.
```

**Train:**
```
Please run train_model with train_path='preprocessed_data/pneumoniamnist/train_loader.pt',
test_path='preprocessed_data/pneumoniamnist/test_loader.pt',
image_size=(28, 28), max_epochs=10, model_name='pneumonia_ddpm'.
```

**Generate:**
```
Please run generate_samples with n_samples=64,
samples_name='pneumonia_samples',
model_checkpoint_path='trained_models/pneumonia_ddpm.ckpt'.
```

**Evaluate:**
```
Please run evaluate_samples with
gen_images_path='generated_samples/pneumonia_samples.pt',
n_real_samples=128, num_permutations=1000.
```

**Run the whole pipeline:**
```
Please run run_pipeline with medmnist_flag='pneumoniamnist',
max_epochs=10, n_samples=64, num_permutations=500, run_name='demo'.
```

## Technical details

### The diffusion model

DDPM (Ho et al., 2020) with:
- Cosine variance schedule (Nichol & Dhariwal, 2021) for smoother noise transitions.
- ShuffleNet-v2 backbone (Ma et al., 2018) — channel-split, depthwise convolution, and channel-shuffle keep parameters low.
- Exponential Moving Average of model weights for more stable generation.
- Clipped reverse diffusion that constrains predicted x₀ to [-1, 1] during sampling.

Samples are produced in the model's native [-1, 1] range; the report task rescales them for display.

### The evaluation method

Rather than FID (which requires a pretrained Inception network and large sample sizes), we use a **patch-based Maximum Mean Discrepancy** test:

1. Extract `p × p` patches from both real and generated images via convolution.
2. Compute pairwise squared L2 distances in patch space.
3. Apply a Gaussian kernel with bandwidth `σ = √median(d²)` (median heuristic on real data only).
4. Compute MMD² with the unbiased estimator (off-diagonal in `K_xx`, `K_yy`).
5. Build the null distribution by pooling real + fake, shuffling, and re-splitting `N` times.
6. p-value is `(1 + #{null ≥ observed}) / (1 + N)` (Phipson–Smyth correction so p never reports as exactly zero).

The `tests/test_permutation.py` script verifies this:
- Under H0 (same distribution) the test does not reject.
- Under H1 (different distributions) the test rejects.
- Under H0, p-values are roughly uniform on [0, 1] (Kolmogorov–Smirnov passes).

### Notes on the v3 rewrite

The previous evaluation had four issues that this version fixes:

- **Single observed statistic.** A two-sample permutation test produces *one* observed value (real vs. generated), not a "distribution" of within-group permutations. Within-group reordering does not change MMD.
- **One-sided p-value.** MMD² ≥ 0 by construction; `P(null ≥ obs)` is the right quantity. The earlier code used absolute values.
- **Bandwidth.** The Gaussian kernel takes `exp(-d² / (2σ²))` where `d` is an L2 *distance*. The earlier `estimate_sigma` returned the median *squared* distance, which made σ far too large and saturated the kernel. We now use `σ = √median(d²)`.
- **Real-sample collection.** `next(iter(loader))` rebuilds the iterator each call, so the previous loop returned the same first batch repeatedly. We now drain the loader once.

## Requirements

- Python 3.10+
- PyTorch 2.3+
- PyTorch Lightning 2.3+
- MedMNIST 3+
- Ollama with Llama 3.2 (or any compatible model) for local mode

See `requirements.txt` for the full list.

## Citations

- Ho, J., Jain, A., Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
- Nichol, A., Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML.
- Ma, N. et al. (2018). *ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design*. ECCV.
- Gretton, A. et al. (2012). *A Kernel Two-Sample Test*. JMLR.
- Phipson, B., Smyth, G. K. (2010). *Permutation P-values Should Never Be Zero*. Statistical Applications in Genetics and Molecular Biology.
- Yang, J. et al. (2023). *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Scientific Data.

## License

MIT
