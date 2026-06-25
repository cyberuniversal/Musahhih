# Musahhih

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALBA7OOTH-Research-Lab/Musahhih/blob/main/notebooks/01_nahw_baseline_unsloth.ipynb)

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

### Run the untouched-model baseline

1. Open [`notebooks/01_nahw_baseline_unsloth.ipynb`](notebooks/01_nahw_baseline_unsloth.ipynb) with the Colab badge above.
2. In Colab, select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run the notebook from top to bottom. The default pilot processes exactly the first 25 test records; the separate 511-record section is disabled by default.
4. Inspect the manual-review table, then download the files from `outputs/` or optionally copy them to Google Drive.

Free Colab GPU availability is not guaranteed. The workflow does not require Colab Pro, paid APIs, or paid storage.

> **Test-only warning:** Never train on Nahw-Passage or use its results to tune prompts, choose checkpoints, or make repeated model-selection decisions.

The pilot writes `outputs/baseline_pilot_predictions.jsonl` and `outputs/baseline_pilot_summary.json`. Generated outputs are ignored by Git.

## Repository structure

```text
.
├── data/
│   └── train.sample.jsonl
├── docs/
│   ├── collaboration_workflow.md
│   ├── dataset_audit.md
│   ├── experiment_naming.md
│   ├── literature_matrix.md
│   ├── papers.md
│   ├── prompt_baseline_protocol.md
│   └── research_plan.md
├── notebooks/
│   └── 01_nahw_baseline_unsloth.ipynb
├── scripts/
│   ├── download_nahw.py
│   ├── inspect_nahw.py
│   ├── nahw_baseline_utils.py
│   ├── prepare_qalb_manifests.py
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

### QALB private manifests

Registered QALB users may place the unchanged release ZIP at the ignored path `data/raw/qalb/QALB-0.9.1-Dec03-2021-SharedTasks.zip`, then run:

```bash
python scripts/prepare_qalb_manifests.py
```

The script reads the ZIP directly and writes corpus-text-free metadata and hashes under ignored `data/processed/qalb/`. It preserves within-split duplicates, excludes train/dev records with exact source overlap against QALB test or Nahw, and keeps every QALB test record evaluation-only. Never commit or redistribute the QALB release or these private outputs.

### B1-P1/B2-P1 prompt baseline scaffolding

The frozen prompt-only baselines are implemented as public scaffolding with private
data kept out of Git:

```bash
python -m unittest tests.test_baseline_prompts tests.test_prepare_b1_prompt_bundle tests.test_run_prompt_baseline -q
```

Licensed QALB users can generate the private B1 demonstration bundle only after
creating the text-free QALB manifests above:

```bash
python -m scripts.prepare_b1_prompt_bundle
```

The B1 bundle is text-bearing and is written under ignored `data/processed/qalb/`.
Do not print, commit, attach, or redistribute it. The command fails closed unless
the frozen structural checks match: 3,116 candidate annotations, 458 distinct
records, and selected identity SHA-256
`76edd4c3de4b6cb5a985464faa066dea40faf9b25b8fa2912b3bf9c4750a9e8c`.
By default, the bundle writer refuses output paths outside `data/processed/qalb/`;
the override flag is only for temporary local diagnostics and must not be used
for committed artifacts.

Canonical output directories for prompt-baseline runs use:

```text
outputs/<experiment-id>/predictions.jsonl
outputs/<experiment-id>/summary.json
outputs/<experiment-id>/run.log
```

`scripts/run_prompt_baseline.py` refuses to overwrite an existing run directory
and refuses `nahw-passage` unless `--confirm-final-eval` is passed deliberately.
Use QALB development for technical validation before any final Nahw-Passage run.

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
The registered run IDs are defined in
[`docs/experiment_naming.md`](docs/experiment_naming.md), and the frozen B0/B1/B2
prompt protocols are defined in
[`docs/prompt_baseline_protocol.md`](docs/prompt_baseline_protocol.md).

## Team workflow

Use the Musahhih Research Hub in Notion for roadmap and status, and GitHub issues
and pull requests for execution. Each active task should have one owner, one
branch, and one PR so human contributors and AI agents do not overwrite each
other. See [`docs/collaboration_workflow.md`](docs/collaboration_workflow.md).

## Foundation

This project builds on:

- [Nahw: A Comprehensive Benchmark of Arabic Grammar Understanding, Error Detection, Correction, and Explanation](https://aclanthology.org/2026.eacl-long.296/)
- [Official Nahw repository](https://github.com/qcri/nahw-arabic-grammar-benchmark)

## Status

Early research and baseline implementation. No model-performance claims are made yet.
