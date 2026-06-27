# Prompt-Baseline Inference Core Design

## Goal

Complete the public execution path for frozen B1-P1 and B2-P1 prompt baselines
without running private QALB data or final-test Nahw-Passage in this task.

## Current gap

`scripts/run_prompt_baseline.py` currently validates an experiment ID, creates a
canonical directory, and writes a `planned` summary. It does not load records,
render the frozen prompts, invoke a model, capture raw output, apply the accepted
parser, write predictions, or finalize artifact hashes. That prevents the
QALB-development technical-validation gate from being executed reproducibly when
a licensed runtime becomes available.

## Chosen approach

Extend the existing runner rather than creating a notebook-only or
Unsloth-specific implementation. Preserve planned-scaffold behavior as the
default and require `--execute` before model loading or generation. Keep heavy
PyTorch/Transformers imports inside the real backend so unit tests and planning
mode remain lightweight.

The execution core accepts an injected callable in tests. The CLI supplies a
lazy Gemma 3 backend in a real runtime. This separates deterministic artifact
logic from GPU/model concerns without adding a new framework.

## Private input contract

Execution input is UTF-8 JSONL. Each non-empty line must contain:

- `record_id`: non-empty string, unique within the file;
- `passage`: string preserved byte-for-byte after JSON decoding;
- `error`: non-empty string preserved without normalization;
- `gold_correction`: optional string used only for development diagnostics;
- `metadata`: optional JSON object copied to the private prediction artifact.

The runner never prints input rows or rendered prompts. Paths inside the
repository are accepted only under ignored private roots (`data/processed/` or
`outputs/`); paths outside the repository are permitted for ephemeral private
runtimes.

For B1-P1, `--bundle` is required. The existing private bundle schema must have
`schema_version: 1` and exactly five demonstrations with string `source`, `error`,
and `correction` values. B2-P1 rejects `--bundle` to prevent misleading metadata.

## Execution flow

1. Validate the experiment ID, final-test confirmation, private paths, input
   schema, and B1 bundle before creating a run directory.
2. In planned mode, create the canonical directory and existing corpus-text-free
   scaffold only.
3. In execution mode, create the run directory exactly once, instantiate the
   backend, render each prompt through `baseline_prompts.py`, generate a raw
   response, and parse it through `nahw_baseline_utils.parse_model_response`.
4. Write one private prediction row per input record containing the record ID,
   metadata, full rendered prompt, prompt SHA-256, raw response, parsed
   correction, parser warnings, optional gold correction, and strict exact-match
   result when gold is present.
5. Write a corpus-text-free `summary.json` containing immutable configuration,
   model ID/revision, deterministic decoding settings, record/diagnostic counts,
   input/bundle/prediction hashes, an aggregate prompt-hash digest, runtime
   metadata, and status `complete`.
6. If execution fails after directory creation, retain the directory and any
   completed prediction rows, write status `invalid`, and record only the error
   class plus a generic non-text-bearing message. Never delete or overwrite the
   run.

## Real backend

The built-in backend uses the same Gemma 3 processor/model family as the B0
runner. It loads `Gemma3ForConditionalGeneration` and `AutoProcessor` lazily,
uses the chat template, performs greedy generation (`do_sample=False`), decodes
only newly generated tokens, and records model revision, maximum new tokens,
device/dtype, and available package versions. No fallback model or decoding
change is selected automatically.

## Output safety

The canonical layout remains:

```text
outputs/<experiment-id>/predictions.jsonl
outputs/<experiment-id>/summary.json
outputs/<experiment-id>/run.log
```

The default output root is Git-ignored. A custom root inside the repository is
rejected unless it remains under `outputs/`. An explicit
`--allow-outside-private-output` override is available only for temporary tests
or diagnostics, matching the existing private B1 bundle pattern. Existing run
directories always fail closed.

## Testing

Synthetic English fixtures avoid committing Arabic corpus-like text. Tests will
prove:

- input schema and unique-ID validation fail before output creation;
- B1 requires exactly five valid demonstrations and B2 rejects a bundle;
- injected generation captures prompts, hashes, raw responses, parsed output,
  warnings, strict exact match, and a text-free summary;
- unsafe output paths and overwrites are rejected;
- a backend failure preserves an `invalid` run without leaking fixture text into
  the summary or log;
- CLI help remains importable without PyTorch/Transformers model loading; and
- all existing repository tests remain green.

## Research safeguards and exclusions

- Frozen B1-P1/B2-P1 templates and B1 selection rules are imported unchanged.
- Nahw-Passage still requires explicit final-evaluation confirmation.
- This task performs no QALB or Nahw inference and reports no model metric.
- It does not prepare transformed QALB development data, select examples,
  revise prompts, choose checkpoints, or change experiment protocols.
- Private inputs, bundles, prompts, raw responses, predictions, credentials,
  weights, checkpoints, and adapters remain out of Git and public logs.
- Parser checks are technical validation, not expert Arabic linguistic review.
