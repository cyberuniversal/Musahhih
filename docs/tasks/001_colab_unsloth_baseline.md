# Task 001 — Colab + Unsloth Nahw Baseline

## Goal

Create a Google Colab notebook that lets a team member run Musahhih's first reproducible baseline without paying for compute.

The notebook must use a free Colab GPU when available, prepare the held-out Nahw-Passage benchmark, load an untouched open-weight model through the current supported Unsloth workflow, run a 25-record pilot, and save predictions plus a summary metric.

## Required output

Create:

```text
notebooks/01_nahw_baseline_unsloth.ipynb
```

Update `README.md` with:

- an **Open in Colab** badge
- brief notebook instructions
- the fact that free GPU availability is not guaranteed

## Notebook flow

1. Display the notebook's purpose and research rules.
2. Verify that a CUDA GPU is available and print its name and memory.
3. Clone or update `https://github.com/cyberuniversal/Musahhih`.
4. Install pinned compatible versions of Unsloth and required libraries using the current official installation method.
5. Print package versions for reproducibility.
6. Run the existing Nahw download, inspection, and evaluation-preparation scripts.
7. Assert that the prepared evaluation file contains 511 records and that it is marked test-only.
8. Load a free-Colab-compatible 4-bit model using Unsloth. Prefer a Gemma 3 4B instruct checkpoint for direct comparison with Nahw when currently supported; otherwise document and use the closest justified open alternative.
9. Run deterministic inference on the first 25 records.
10. Extract only the corrected token without changing Arabic diacritics or spelling beyond documented whitespace/quote normalization.
11. Save every prompt, raw response, parsed response, gold correction, and exact-match result.
12. Write a JSON summary containing model ID, revision if available, sample count, correct count, exact-match accuracy, decoding settings, package versions, GPU, and timestamp.
13. Provide optional cells for downloading the result files or copying them to Google Drive.
14. Keep the full 511-record run in a separate disabled/manual section after the pilot.

## Research constraints

- Never train on Nahw-Passage.
- Do not fine-tune in this notebook yet.
- Do not use the 25 test examples to optimize the prompt repeatedly.
- Do not report a score unless the notebook actually produced it.
- Do not require Colab Pro, paid APIs, or paid storage.
- Do not commit secrets, model weights, datasets, or large generated output files.

## Engineering requirements

- Reuse repository scripts where practical instead of duplicating their logic.
- Keep cells runnable from a fresh Colab session in top-to-bottom order.
- Make installation and repository setup cells safe to rerun where practical.
- Use clear Markdown headings and short explanations for non-expert team members.
- Fail early with a useful message when no GPU is assigned.
- Pin or record package versions so later runs can be compared.

## Acceptance criteria

- The notebook is valid `.ipynb` JSON and opens in Colab.
- Setup, data preparation, model loading, inference, parsing, and evaluation are separate sections.
- The default run processes exactly 25 test records.
- Predictions and summary metrics are written to clearly named files under `outputs/` during runtime.
- No test record is used for training.
- README contains a working Colab link and accurate usage instructions.
- `python -m compileall scripts` passes.
- The final response states what was actually executed versus only statically validated.

## Do not do yet

- Do not add QALB or Tibyan training code.
- Do not launch QLoRA training.
- Do not redesign the whole repository.
- Do not claim that the baseline outperforms any published system.
