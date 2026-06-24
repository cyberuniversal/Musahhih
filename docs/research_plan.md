# Experimental Plan

## Final question to lock first

**Does LoRA/QLoRA supervised fine-tuning on natural, synthetic, or mixed Arabic GEC data improve an open model's MSA correction accuracy over the untouched model and prompt-only baselines?**

## Hypotheses

- H1: Fine-tuning will outperform zero-shot and few-shot prompting on held-out Arabic GEC data.
- H2: Natural expert-written/validated data will be more sample-efficient than synthetic data.
- H3: Mixed natural + synthetic training will outperform synthetic-only training.
- H4: Targeted GEC fine-tuning may cause overcorrection or capability loss, so both must be measured.

## Phase A — Reproducible baseline

Model:
- First choice for direct connection to Nahw: `google/gemma-3-4b-it`
- A text-only causal LM may be added later for easier LoRA experiments.

Test set:
- Nahw-Passage, held out completely.

Baselines:
- B0-P1: untouched model, zero-shot
- B1-P1: untouched model, five-shot with deterministically selected eligible QALB train examples
- B2-P1: untouched model, explicit expert-style correction prompt with no demonstrations

The exact frozen prompts, demonstration-selection rule, and pre-test validation gate
are defined in [`prompt_baseline_protocol.md`](prompt_baseline_protocol.md). Run and
artifact identifiers follow [`experiment_naming.md`](experiment_naming.md). B1 is
the few-shot family and B2 is the expert-style family; do not reverse these labels.

Primary metric:
- exact correction accuracy on the highlighted erroneous token, matching Nahw's GEC setup

Secondary diagnostics:
- normalized exact match
- empty/invalid response rate
- overlong response rate
- performance by passage and correction form

## Phase B — Training data

Do not invent linguistic labels.

Potential training sources:
- QALB train/dev splits, if legally obtained
- expert-validated public corpora
- Tibyan, only after confirming the released data and license
- synthetic data released by the relevant studies, if compatible

Unified training record:

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "صحح الكلمة الخاطئة المحددة في النص التالي، وأعد الكلمة المصححة فقط.\nالنص: ...\nالكلمة الخاطئة: ..."
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "الكلمة المصححة"
    }
  ],
  "source": "dataset_name",
  "split": "train"
}
```

## Phase C — Fine-tuning experiments

Hold the following constant:
- base model
- validation/test data
- prompt format
- random seeds where practical
- decoding parameters
- evaluation script

Runs:
- F1: natural-only
- F2: synthetic-only, matched for sample count
- F3: natural + synthetic
- F4: best mixture with data-size ablation

## Phase D — Evaluation

Evaluate all systems on:
- Nahw-Passage GEC
- QALB official test split, if licensed
- another held-out corpus if compatible

Compare:
- B0, B1, B2
- F1, F2, F3, F4

Also measure:
- unchanged/correct input behavior to estimate overcorrection
- ArabicMMLU or another general Arabic benchmark before and after fine-tuning
- inference cost and adapter size

## Phase E — Paper contribution

A credible paper should contribute:
1. A controlled extension of Nahw from GU fine-tuning to actual GEC fine-tuning.
2. A natural-versus-synthetic-versus-mixed comparison.
3. Reproducible open adapters and evaluation code.
4. Error/capability-retention analysis.
5. Clear limitations: no claim of expert-level Arabic and no manual linguistic annotation by non-linguists.

## Go/no-go rule after the pilot

Continue to the full study only if:
- baseline evaluation runs reproducibly
- at least one training corpus is legally available
- the fine-tuned pilot improves held-out accuracy without obvious leakage
