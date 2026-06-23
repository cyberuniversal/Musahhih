# AGENTS.md

## Purpose

Musahhih is a research repository for improving Modern Standard Arabic grammatical error correction with open-weight language models.

Treat this file as a map, not a full manual. Follow the linked docs for details.

## Sources of truth

- Project overview and setup: `README.md`
- Experimental design: `docs/research_plan.md`
- Dataset roles and leakage rules: `docs/dataset_audit.md`
- Prior work and research gap: `docs/literature_matrix.md`
- Current implementation task: `docs/tasks/001_colab_unsloth_baseline.md`

## Current milestone

Build and validate a free Google Colab/Jupyter workflow using Unsloth that runs an untouched-model baseline on 25 held-out Nahw-Passage records.

Do not begin fine-tuning until the baseline and parser are verified and a legally usable GEC training corpus is available.

## Non-negotiable research rules

- `Nahw-Passage` is test-only. Never train, tune prompts, or select checkpoints on it.
- Preserve official train/dev/test splits for every external dataset.
- Do not invent Arabic linguistic labels or claim expert validation.
- Never fabricate metrics, dataset access, completed runs, or citations.
- Record exact model IDs/revisions, prompts, decoding settings, seeds, hardware, and package versions.
- Save predictions as well as aggregate metrics.
- Prefer zero-cost tools and Google Colab Free; do not introduce paid dependencies.
- Never commit API keys, Hugging Face tokens, Google credentials, or private datasets.

## Repository conventions

- Put reusable Python code in `scripts/`.
- Put Colab/Jupyter notebooks in `notebooks/`.
- Put small, human-readable result summaries in `results/`.
- Keep downloaded data, checkpoints, adapters, and large outputs out of Git.
- Use UTF-8 for Arabic text and preserve original strings unless normalization is part of an explicitly documented metric.
- Keep changes narrow. Do not refactor unrelated files.

## Validation

For Python changes, run:

```bash
python -m compileall scripts
```

For data preparation, run:

```bash
python scripts/download_nahw.py
python scripts/inspect_nahw.py
python scripts/prepare_nahw_eval.py
```

For notebooks:

- validate that the notebook is valid JSON
- make setup cells idempotent where practical
- ensure a fresh Colab runtime can run cells in order
- keep the 25-example pilot separate from the full 511-record run

If a required check cannot run because no GPU or external access is available, state that clearly and report what was validated instead.

## Working style

- Read the relevant docs before editing.
- Inspect existing files before creating replacements.
- For tasks expected to take multiple hours, create or update `PLANS.md` before implementation.
- Prefer a small working vertical slice over a broad unfinished system.
- End each task with a concise summary of changed files, checks run, unresolved issues, and the next step.
