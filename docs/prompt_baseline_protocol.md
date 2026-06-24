# Prompt-Only Baseline Protocol

Status: B0-P1 accepted; B1-P1 and B2-P1 frozen on 2026-06-24 but not yet run.

This document preregisters the untouched-model prompt baselines before either new
protocol is evaluated on Nahw-Passage. It preserves the experiment-family mapping
already established in `docs/research_plan.md`:

- `B0-P1`: zero-shot minimal instruction;
- `B1-P1`: five-shot in-context examples;
- `B2-P1`: zero-shot expert-style instruction.

All three protocols use the same untouched model, processor, parser, deterministic
decoding settings, and Nahw-Passage records. No adapters are attached and no model
weights are trained or modified.

## Fixed inference configuration

Unless a documented compatibility failure requires a new protocol revision:

- model: `unsloth/gemma-3-4b-it-unsloth-bnb-4bit`;
- model revision: record the immutable revision resolved at execution time;
- quantization: Unsloth 4-bit loading;
- seed: `3407`;
- decoding: greedy, `do_sample=false`;
- temperature: not passed;
- maximum new tokens: `32`;
- parser: `scripts.nahw_baseline_utils.parse_model_response`;
- evaluation: exact match after stripping outer whitespace from the gold value
  only, with no Arabic-letter, hamza, diacritic, or spelling normalization.

The exact package versions, processor class, CUDA version, GPU, Git SHA, prompt,
and input checksum must be captured in each run summary.

## B0-P1: accepted zero-shot baseline

The exact template is the `PROMPT` constant in `scripts/prepare_nahw_eval.py`:

```text
صحح الكلمة الخاطئة المحددة في النص التالي.
أعد الكلمة المصححة فقط دون شرح أو علامات اقتباس.

النص:
{passage}

الكلمة الخاطئة:
{error}
```

B0-P1 has already been executed and audited. Its accepted result is documented in
`results/b0_full_baseline_audit.md`. Do not rerun it to guide B1 or B2 design.

## B1-P1: frozen five-shot baseline

B1-P1 adds five fixed demonstrations before the unchanged B0 query. Demonstration
content must come only from the eligible QALB-2014 L1 training selection. QALB
development and all QALB test records are forbidden as demonstrations.

The exact assembled template is:

```text
فيما يلي خمسة أمثلة على المهمة نفسها. في كل مثال، أعدت الأداة الكلمة المصححة فقط.

المثال 1:
النص:
{demo_1_passage}
الكلمة الخاطئة:
{demo_1_error}
الكلمة المصححة:
{demo_1_correction}

المثال 2:
النص:
{demo_2_passage}
الكلمة الخاطئة:
{demo_2_error}
الكلمة المصححة:
{demo_2_correction}

المثال 3:
النص:
{demo_3_passage}
الكلمة الخاطئة:
{demo_3_error}
الكلمة المصححة:
{demo_3_correction}

المثال 4:
النص:
{demo_4_passage}
الكلمة الخاطئة:
{demo_4_error}
الكلمة المصححة:
{demo_4_correction}

المثال 5:
النص:
{demo_5_passage}
الكلمة الخاطئة:
{demo_5_error}
الكلمة المصححة:
{demo_5_correction}

الآن نفذ المهمة على النص التالي.
صحح الكلمة الخاطئة المحددة في النص التالي.
أعد الكلمة المصححة فقط دون شرح أو علامات اقتباس.

النص:
{passage}

الكلمة الخاطئة:
{error}
```

### Demonstration-selection rule

The implementation must read the unchanged licensed QALB archive and the verified
private training-selection manifest. It must select records using this frozen rule:

1. Use only QALB release `0.9.1`, year `2014`, track `L1`, split `train`, with
   `eligible_for_training=true`.
2. Parse the corresponding M2 block without normalizing its UTF-8 Arabic text.
3. Treat each annotation as a candidate. Retain only `Edit` annotations that
   replace exactly one whitespace-delimited source token with exactly one non-empty
   whitespace-delimited correction token. Other annotations in the same record do
   not disqualify that candidate.
4. Require the erroneous token to occur exactly once in the source token sequence,
   require the correction to differ from it, and require a source length of 5 to
   40 tokens inclusive.
5. Give each candidate the identity `record_key + "|" + start + ":" + end`.
   Compute `SHA-256(candidate_identity + "|B1-P1")` as UTF-8 and sort candidates
   ascending by that digest. Walk the sorted list, taking the first candidate from
   each distinct record until five records have been selected.
6. Preserve the source, erroneous token, and correction exactly. Do not alter
   letters, diacritics, punctuation, or spelling.
7. Store the private selected record keys, text, source hashes, prompt-bundle hash,
   and selection-manifest hash under ignored `data/processed/qalb/`. Do not commit
   or redistribute that bundle.

This deterministic content-neutral rule prevents manual selection of examples that
look favorable. If fewer than five records satisfy it, stop and register a revised
rule before viewing any Nahw result.

The verified local dry run against the registered QALB 0.9.1 archive produced
3,116 candidate annotations across 458 distinct records and selected five. The
SHA-256 of the five selected candidate identities joined by `\n`, in prompt order,
was `76edd4c3de4b6cb5a985464faa066dea40faf9b25b8fa2912b3bf9c4750a9e8c`.
This is an identity check, not a corpus-text hash or a model result.

Five demonstrations were chosen before evaluation because the primary prompting
study reports stronger five-shot than one-shot results in its setting. That result
does not establish that five-shot will improve Gemma or Nahw-Passage; B1-P1 tests
that transfer rather than assuming it. The study's chain-of-thought answer format
is not copied because Musahhih requires a single corrected token and conservative
automatic parsing.

## B2-P1: frozen expert-style baseline

B2-P1 uses no examples and does not list or invent linguistic error labels. The
word "expert-style" describes the prompting strategy; it is not a claim that the
prompt or outputs received expert linguistic validation.

The exact template is:

```text
أنت أداة متخصصة في تصحيح العربية الفصحى.
راجع سياق النص لتحديد الصيغة الصحيحة للكلمة المحددة، مع إبقاء بقية النص دون تغيير.
أعد الكلمة المصححة فقط دون شرح أو تعليل أو علامات اقتباس.

النص:
{passage}

الكلمة الخاطئة:
{error}
```

This is a deliberately compact adaptation of expert prompting. It assigns a clear
role and procedure but avoids the source paper's broad error taxonomy because the
Nahw task supplies the erroneous word and this project does not create linguistic
labels.

## Pre-test validation gate

Before either protocol touches Nahw-Passage:

1. Implement prompt assembly and unit-test exact whitespace and ordering.
2. Reproduce the five-example B1 private bundle twice and verify identical hashes.
3. Confirm every B1 demonstration is selected from eligible QALB train and no demo
   belongs to QALB dev, QALB test, or Nahw-Passage.
4. Use eligible QALB development records only for implementation checks such as
   model loading, prompt length, response capture, parser behavior, and output-file
   integrity. Do not change the frozen prompt based on Nahw-Passage.
5. Review a small QALB-development output sample for technical formatting only.
   Do not claim expert linguistic validation.
6. Record the model revision, prompt hashes, demonstration-bundle hash, parser
   version, decoding settings, and Git SHA before the final evaluation.

If validation reveals a technical defect, mark the affected run invalid, document
the defect, and create `P2`. Do not overwrite `P1` or silently edit its template.

## Final evaluation rule

After the gate passes, run B1-P1 and B2-P1 once each on all 511 Nahw-Passage
records. Save separate raw predictions and summaries under their canonical
experiment IDs. Do not use either final result to tune the prompt, demonstrations,
parser, checkpoint, or model. Any later exploratory protocol must be labeled
separately and cannot replace the preregistered comparison.

## Sources

- [Beyond English: Evaluating LLMs for Arabic Grammatical Error Correction](https://aclanthology.org/2023.arabicnlp-1.9/)
- [Nahw: A Comprehensive Benchmark of Arabic Grammar Understanding, Error Detection, Correction, and Explanation](https://aclanthology.org/2026.eacl-long.296/)
- `docs/dataset_audit.md` for the QALB license, split, overlap, and checksum rules.
