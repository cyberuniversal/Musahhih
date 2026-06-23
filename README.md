# Musahhih

**Musahhih** is an open research project on improving Modern Standard Arabic grammatical error correction with open-weight language models.

The project builds on the **Nahw** benchmark and related Arabic grammatical error correction work. Its first goal is to establish a reproducible baseline on held-out Arabic correction data, then compare LoRA/QLoRA fine-tuning with natural, synthetic, and mixed training corpora.

## Research question

> Can supervised fine-tuning of an open model on existing Arabic grammatical-error-correction data improve MSA correction performance over the untouched model and prompt-only baselines?

## Current scope

Primary task:

- **Grammatical Error Correction (GEC)**

Possible later extensions:

- Grammatical Error Detection (GED)
- Grammatical Error Explanation (GEX), if qualified linguistic evaluation becomes available

The project does not create new linguistic labels without qualified Arabic linguists. It relies on existing expert-written or expert-validated datasets.

## First milestone

Produce a reproducible baseline score from an untouched open model on the held-out **Nahw-Passage** benchmark.

> **Data rule:** `Nahw-Passage` is treated as evaluation data. It must not be used for training if results are reported on it.

## Repository structure

```text
.
├── data/
│   └── train.sample.jsonl
├── docs/
│   ├── dataset_audit.md
│   ├── literature_matrix.md
│   ├── papers.md
│   └── research_plan.md
├── scripts/
│   ├── download_nahw.py
│   ├── inspect_nahw.py
│   ├── prepare_nahw_eval.py
│   ├── run_gemma3_nahw_baseline.py
│   └── train_lora.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and prepare the benchmark:

```bash
pip install -r requirements.txt
python scripts/download_nahw.py
python scripts/inspect_nahw.py
python scripts/prepare_nahw_eval.py
```

Authenticate with Hugging Face if the selected model is gated:

```bash
huggingface-cli login
```

Run a small baseline:

```bash
python scripts/run_gemma3_nahw_baseline.py --limit 25
```

Predictions and metrics are written to `outputs/`.

## Planned experiments

1. Untouched-model zero-shot baseline
2. Prompt-only baselines
3. Natural-data LoRA/QLoRA fine-tuning
4. Synthetic-data LoRA/QLoRA fine-tuning
5. Mixed natural + synthetic fine-tuning
6. Held-out GEC evaluation
7. General Arabic capability-retention checks

See [`docs/research_plan.md`](docs/research_plan.md) for the full experimental design.

## Foundation

This project builds on:

- [Nahw: A Comprehensive Benchmark of Arabic Grammar Understanding, Error Detection, Correction, and Explanation](https://aclanthology.org/2026.eacl-long.296/)
- [Official Nahw repository](https://github.com/qcri/nahw-arabic-grammar-benchmark)

## Status

Early research and baseline implementation. No model-performance claims are made yet.