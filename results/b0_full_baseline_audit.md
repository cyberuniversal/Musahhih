# B0 full untouched-model baseline audit

Date reviewed: 2026-06-23

## Accepted result

The supplied full-run artifacts contain 511 unique Nahw-Passage correction records and match the prepared test-file identity and notebook protocol. The run produced 86 exact matches:

- exact-match accuracy: `86 / 511 = 0.16829745596868884` (16.83%)
- invalid or empty responses: 0
- parsing failures: 0
- suspicious outputs: 7
- multi-token outputs: 7

This is an actually executed model result, not an estimate. It is the untouched 4-bit `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` baseline on a Tesla T4; no adapters were attached and no fine-tuning occurred.

## Artifact identity

- predictions SHA-256: `6997b6fe5959f5502511ebdd1885d05a89ebaefeb27eefb73520842598f36ebc`
- summary SHA-256: `93294aa8dc472f8e824ff0d39b4b2ab76d3454f46a29c72508d2f3e1d2f6ea0d`
- prepared test SHA-256: `acb3cfd204b35d5415532fbd32a4a5231b553fae329ab8f48e8454609e10279b`
- repository commit recorded by the run: `8735c29b48fbc45c23c9aff4528ee831aa2bcd1c`
- model revision: `316726ca0bd24aa323bfaf86e8a379ee1176d1fe`
- decoding: greedy (`do_sample=false`), temperature not passed, `max_new_tokens=32`, seed 3407

The complete generated files are stored locally under ignored `outputs/` and are not committed.

## Parser review

All seven responses carrying substantive warnings were manually inspected at the output-format level. Every one was an incorrect prediction. The parser retained multi-line or multi-word content instead of altering Arabic spelling or selecting an answer using the gold label, so these warnings produced no false-positive exact matches.

This was a technical parser/scorer review, not expert Arabic linguistic validation.

## Reproducibility caveat

One record shared by the 25-example pilot and the full run produced different raw responses even though both summaries record the same model revision, packages, prompt, seed, hardware class, and greedy decoding settings. Both responses were incorrect, so the reported score is unaffected. The discrepancy means byte-identical generation is not yet demonstrated and should remain visible in future comparisons.

## Decision

B0 is frozen as the project's untouched-model baseline with the 16.83% exact-match result and the reproducibility caveat above. Nahw-Passage remains test-only. Do not use its results to tune prompts, choose checkpoints, select training examples, or make repeated model decisions.

The experiment naming convention was registered after this run. Its retrospective
canonical ID is `B0-P1__gemma3-4b-it__nahw-passage__s3407__r01`. This alias does not
rename or modify the accepted artifacts and does not indicate a rerun.
