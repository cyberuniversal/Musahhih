# Experiment Naming Convention

Status: frozen on 2026-06-24.

This convention gives every model run one stable, file-safe identity. It applies to
baseline, fine-tuning, validation, and final evaluation runs. A run ID identifies
the protocol family, model, evaluation dataset, seed, and replicate. Exact model
revisions, data hashes, prompts, decoding settings, hardware, and package versions
belong in the run summary rather than in the filename.

## Canonical experiment ID

Use:

```text
<family>-<protocol>__<model>__<evaluation>__s<seed>__r<replicate>
```

Example:

```text
B1-P1__gemma3-4b-it__nahw-passage__s3407__r01
```

Rules:

- `family` and `protocol` use the registered values below.
- `model` and `evaluation` use lowercase ASCII letters, digits, periods, and
  hyphens only.
- `seed` is the integer supplied to the inference or training stack.
- `replicate` is two digits starting at `01`.
- Do not put dates, accuracy values, people, GPU names, or words such as `final`,
  `new`, or `best` in an experiment ID.
- Never reuse an experiment ID for a different configuration. A rerun with the
  same configuration increments the replicate.

The accepted pattern is:

```regex
^(B[0-2]|F[1-4])-P[0-9]+__[a-z0-9][a-z0-9.-]*__[a-z0-9][a-z0-9.-]*__s[0-9]+__r[0-9]{2}$
```

## Registered families

| Family | Meaning | Current protocol |
|---|---|---|
| `B0` | Untouched zero-shot model | `B0-P1` |
| `B1` | Untouched few-shot model | `B1-P1` |
| `B2` | Untouched expert-style prompted model | `B2-P1` |
| `F1` | Natural-data fine-tuning | Not yet registered |
| `F2` | Synthetic-data fine-tuning | Not yet registered |
| `F3` | Mixed natural and synthetic fine-tuning | Not yet registered |
| `F4` | Registered mixture or data-size ablation | Not yet registered |

`B1` remains the few-shot family and `B2` remains the expert-style prompt family.
Do not reverse these labels in dashboards, notebooks, filenames, or papers.

## Registered slugs

Use these values unless a future protocol amendment registers another value:

| Item | Slug |
|---|---|
| Gemma 3 4B Instruct | `gemma3-4b-it` |
| Nahw-Passage | `nahw-passage` |
| QALB-2014 development | `qalb14-dev` |
| QALB-2015 L2 development | `qalb15-l2-dev` |

The frozen B0 run predates this convention. Its canonical retrospective ID is:

```text
B0-P1__gemma3-4b-it__nahw-passage__s3407__r01
```

This alias does not change the recorded artifact hashes or imply that the run was
executed again.

## Artifact layout

Generated artifacts remain ignored by Git:

```text
outputs/<experiment-id>/predictions.jsonl
outputs/<experiment-id>/summary.json
outputs/<experiment-id>/run.log
```

Small, reviewed, corpus-text-free reports may be committed under:

```text
results/<experiment-id>.md
```

Every `summary.json` must record at least:

- experiment ID and run status;
- timestamp in UTC and Git commit SHA;
- model ID and immutable revision when available;
- protocol ID and exact prompt template;
- dataset name, split, input checksum, and selection-manifest checksum;
- decoding or training settings and random seed;
- GPU, Python, CUDA, PyTorch, Transformers, Unsloth, Accelerate, PEFT, and TRL
  versions when applicable;
- prediction artifact checksum;
- metric definitions and aggregate results; and
- any deviation, warning, interruption, or reproducibility caveat.

## Run status

Use exactly one status:

- `planned`: registered but not ready to execute;
- `frozen`: configuration is fixed and ready for implementation or execution;
- `running`: execution started but is not complete;
- `complete`: artifacts passed the required checks;
- `invalid`: a protocol deviation, corruption, leakage risk, or execution failure
  makes the result unusable for comparison.

An invalid run keeps its ID and artifacts for auditability. Never overwrite it
with a replacement run.

## Amendment rule

Changing a prompt, demonstration-selection rule, model, dataset split, metric,
parser, or decoding setting creates a new protocol revision or replicate as
appropriate. Document the reason before running it. Nahw-Passage results must
never be used as the reason for a prompt or checkpoint change.
