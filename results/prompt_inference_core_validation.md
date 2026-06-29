# Prompt-baseline inference core validation

Date: 2026-06-27

GitHub issue: https://github.com/ALBA7OOTH-Research-Lab/Musahhih/issues/8

## Scope

The B1-P1 and B2-P1 runner now supports explicit, revision-pinned Gemma
inference while retaining planned scaffolding as the default. It validates the
private record schema and protocol-specific demonstration contract, renders the
frozen prompts, captures raw and parsed predictions, and writes corpus-text-free
summary metadata. Failed executions preserve partial artifacts and are marked
invalid rather than overwritten.

## Validation

- `python3 -m unittest tests.test_run_prompt_baseline -q`: 15 tests passed.
- `python3 -m unittest discover -s tests -q`: 66 tests passed, 2 skipped.
- `git diff --check`: passed.

Tests use synthetic ASCII fixtures. They cover record and bundle validation,
private path boundaries, B1/B2 prompt execution, parsing and strict exact-match
capture, complete and invalid summaries, partial-write preservation, lazy CLI
execution controls, revision pinning, and planned-mode behavior.

## Research and data safeguards

- No QALB or Nahw corpus text, generated model output, credentials, checkpoints,
  or adapters were used or committed.
- No model inference or performance evaluation was run for this change.
- Nahw-Passage remains test-only and still requires explicit final-evaluation
  confirmation. Nothing here changes a frozen prompt or protocol.
- Private predictions remain under ignored output paths by default. CLI output
  contains only the experiment ID, run directory, and status.

## Remaining operational validation

A licensed contributor with the private QALB development input and gated-model
access must still perform the frozen technical-validation run. That run should
verify model loading on the intended hardware and inspect private artifacts
locally before any Nahw-Passage execution. Results must not be committed if they
contain private text.
