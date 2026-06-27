# Prompt-Baseline Inference Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing planned-run scaffold into an explicit, private-data-safe B1-P1/B2-P1 inference and artifact-capture path.

**Architecture:** Keep pure validation, prompt rendering, prediction capture, and summary logic in `scripts/run_prompt_baseline.py`, with an injected generation callable for tests. Load PyTorch and Transformers only inside a Gemma backend used by `--execute`, while planned mode remains lightweight and backward compatible.

**Tech Stack:** Python standard library, existing prompt/parser helpers, `unittest`, lazy PyTorch/Transformers runtime.

---

### Task 1: Add strict private input and bundle contracts

**Files:**
- Modify: `tests/test_run_prompt_baseline.py`
- Modify: `scripts/run_prompt_baseline.py`

- [ ] **Step 1: Write failing input-loader tests**

Add imports for `load_prompt_records`, `load_b1_demos`, `PromptRecord`, and
`validate_private_path`. Add synthetic tests that write JSONL rows with
`record_id`, `passage`, `error`, optional `gold_correction`, and `metadata`.
Assert exact text preservation, duplicate-ID rejection, malformed-type rejection,
and rejection of repository paths outside `data/processed/` and `outputs/`.

```python
rows = load_prompt_records(input_path)
self.assertEqual(rows[0], PromptRecord("r1", "alpha beta", "beta", "better", {"split": "dev"}))
with self.assertRaisesRegex(RunSafetyError, "duplicate record_id"):
    load_prompt_records(duplicate_path)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_run_prompt_baseline -q
```

Expected: import failure because the new APIs do not exist.

- [ ] **Step 3: Implement the minimal input and bundle APIs**

Add immutable `PromptRecord`, a JSONL loader that validates every field and
unique ID, private-path validation, and a B1 bundle loader that maps exactly five
schema-v1 demonstrations to existing `PromptDemo` objects. Add
`load_protocol_demos(protocol_id, bundle_path)` so B1 requires the bundle and B2
rejects one.

```python
@dataclass(frozen=True)
class PromptRecord:
    record_id: str
    passage: str
    error: str
    gold_correction: str | None
    metadata: dict
```

- [ ] **Step 4: Run focused and full tests and verify GREEN**

Run:

```bash
python3 -m unittest tests.test_run_prompt_baseline -q
python3 -m unittest discover -s tests -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add scripts/run_prompt_baseline.py tests/test_run_prompt_baseline.py
git commit -m "Validate private prompt inference inputs"
```

### Task 2: Execute prompts and preserve complete/invalid artifacts

**Files:**
- Modify: `tests/test_run_prompt_baseline.py`
- Modify: `scripts/run_prompt_baseline.py`

- [ ] **Step 1: Write failing execution tests**

Add an injected deterministic generator and assert B2 prompt rendering, prompt
hash, raw response, conservative parsing, warnings, strict exact match,
prediction SHA-256, aggregate prompt hash, and text-free summary fields. Add a
second generator that raises after one record and assert the run directory,
partial predictions, `invalid` summary, and non-text-bearing log remain.

```python
def generate(prompt: str) -> str:
    return "**fixed**"

summary = execute_run(config, records, [], generate, outputs_root=root)
self.assertEqual(summary["run_status"], "complete")
self.assertNotIn("alpha beta", json.dumps(summary))
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_run_prompt_baseline -q
```

Expected: failure because `execute_run` and prediction finalization do not exist.

- [ ] **Step 3: Implement minimal execution and failure preservation**

Add `render_record_prompt`, `aggregate_prompt_sha256`,
`summarize_prompt_predictions`, atomic JSONL writes with immediate flush, and
`execute_run`. Create the directory once, retain full private prediction rows,
write a corpus-text-free complete summary, and on exception write an invalid
summary plus generic log before re-raising `RunSafetyError`.

The prediction record must contain only these stable fields:

```python
{
    "record_id": record.record_id,
    "metadata": record.metadata,
    "prompt": prompt,
    "prompt_sha256": prompt_sha256(prompt),
    "raw_response": raw,
    "parsed_correction": parsed,
    "parsing_warnings": warnings,
    "gold_correction": record.gold_correction,
    "exact_match": parsed == record.gold_correction if record.gold_correction is not None else None,
}
```

- [ ] **Step 4: Verify GREEN and no-overwrite behavior**

Run:

```bash
python3 -m unittest tests.test_run_prompt_baseline -q
python3 -m unittest discover -s tests -q
```

Expected: all tests pass, including a second execution refusing the existing run.

- [ ] **Step 5: Commit Task 2**

```bash
git add scripts/run_prompt_baseline.py tests/test_run_prompt_baseline.py
git commit -m "Capture prompt baseline inference artifacts"
```

### Task 3: Add explicit CLI execution and lazy Gemma backend

**Files:**
- Modify: `tests/test_run_prompt_baseline.py`
- Modify: `scripts/run_prompt_baseline.py`

- [ ] **Step 1: Write failing CLI/backend-contract tests**

Add a subprocess `--help` smoke test and argument tests proving planning mode is
the default, `--execute` is explicit, B1 requires `--bundle`, B2 rejects it, and
custom private output roots require the deliberate override.

```python
result = subprocess.run(
    [sys.executable, "-m", "scripts.run_prompt_baseline", "--help"],
    check=True,
    text=True,
    capture_output=True,
)
self.assertIn("--execute", result.stdout)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_run_prompt_baseline -q
```

Expected: `--execute` is absent or CLI contract assertions fail.

- [ ] **Step 3: Implement explicit execution and backend**

Add CLI flags `--execute`, `--model`, `--model-revision`,
`--max-new-tokens`, and `--allow-outside-private-output`. Add a
`GemmaGenerator` that imports torch/transformers in its constructor, loads the
exact requested revision, applies the chat template, greedily generates, decodes
new tokens only, and exposes corpus-text-free runtime metadata. Planning mode
must not instantiate the backend or load private records.

- [ ] **Step 4: Verify focused/full tests and compilation**

Run:

```bash
python3 -m unittest tests.test_run_prompt_baseline -q
python3 -m unittest discover -s tests -q
python3 -m compileall -q scripts
```

Expected: all tests and compilation pass without loading a model.

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/run_prompt_baseline.py tests/test_run_prompt_baseline.py
git commit -m "Add opt-in Gemma prompt inference"
```

### Task 4: Document, validate, and publish the scoped task

**Files:**
- Modify: `README.md`
- Create: `results/prompt_inference_core_validation.md`

- [ ] **Step 1: Document the private execution contract**

Update README with the JSONL fields, B1/B2 bundle rules, planned versus
`--execute` behavior, canonical private artifacts, example commands using only
paths/placeholders, invalid-run preservation, and warnings that no QALB/Nahw run
was performed here.

- [ ] **Step 2: Write the corpus-text-free validation report**

Record issue/task IDs, design decisions, files changed, TDD coverage, exact test
commands, checks not run, and research safeguards. Do not include fixture,
private, prompt, or prediction text.

- [ ] **Step 3: Run the final verification gate**

Run:

```bash
python3 -m compileall -q scripts
python3 -m unittest discover -s tests -q
git diff --check origin/main...HEAD
git status --short
```

Also verify only intended files changed, all tracked outputs are corpus-text-free,
no Arabic script appears in the new validation report, and no credential/private
QALB path pattern appears in the diff.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md results/prompt_inference_core_validation.md
git commit -m "Document prompt inference validation"
```

- [ ] **Step 5: Publish and update tracking**

Publish `codex/8-prompt-inference-core`, open a draft PR closing issue #8, update
Notion MSH-15 with branch/commits/tests/risks, and move it to Review. Keep the
worktree for PR feedback and do not merge the PR.
