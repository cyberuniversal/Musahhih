# Tibyan dataset intake

Retrieval date: 2026-06-25

## Recommendation

**Conditionally usable** for synthetic or mixed-data GEC training, after a separate
private manifest task prepares an immutable local manifest and reruns overlap
checks against private QALB data.

The authoritative release was located on Zenodo and the archive was downloadable
without registration. The remaining conditions are methodological rather than
access blockers: no official train/dev/test split is provided, the archive has no
record IDs or error-label fields, the paper-level token statistics do not exactly
match a simple whitespace count over the released final files, and private QALB
overlap cannot be checked in this public branch context.

## Primary sources checked

| Source | Result |
|---|---|
| arXiv paper, `arXiv:2411.04588`, DOI `10.48550/arXiv.2411.04588` | Describes Tibyan, generation/annotation procedure, expert/professional review, ARETA analysis, and states that the corpus is publicly available. The rendered paper and arXiv source package do not include a dataset URL. |
| arXiv source package for `2411.04588` | Confirmed no hidden repository, dataset-card, Zenodo, Hugging Face, or GitHub URL in the LaTeX source. |
| Zenodo record `14623621`, DOI `10.5281/zenodo.14623621` | Located an author archive titled `Tibyan-corpus`; resource type is Dataset; creator listed as `Alrehili, Ahlam`, King Abdulaziz University, ORCID `0000-0002-4218-4659`; license is `cc-by-4.0`; file download succeeded. |
| Hugging Face dataset API search for `Tibyan` | Returned only an unrelated Quran dataset (`Kandil7/tibyan-quran-complete`), not this GEC corpus. |
| GitHub repository search for `Tibyan Arabic GEC` | No matching repository found. |

## Authoritative release metadata

| Field | Value |
|---|---|
| Canonical dataset name | `Tibyan-corpus` on Zenodo; paper title uses “Tibyan Corpus: Balanced and Comprehensive Error Coverage Corpus Using ChatGPT for Arabic Grammatical Error Correction”. |
| Authoritative URL | <https://zenodo.org/records/14623621> |
| Dataset DOI | <https://doi.org/10.5281/zenodo.14623621> |
| Concept DOI | <https://doi.org/10.5281/zenodo.14623620> |
| Publication date | 2025-01-09 |
| Record created | 2025-01-09T16:38:39Z |
| Record modified | 2025-01-10T13:27:31Z |
| Version/tag/commit | No explicit version, tag, or commit in the Zenodo metadata. Use the record DOI plus retrieval date as the immutable release identifier. |
| Access process | Public direct download from Zenodo; no registration or credentials required in this check. |
| License | CC BY 4.0 (`cc-by-4.0`) in Zenodo metadata. Use requires attribution, link to the license, indication of changes, and no added legal/technical restrictions. |

## Archive and checksum verification

| Item | Value |
|---|---|
| Zenodo file name | `Data (1).rar` |
| Zenodo file size | 5,614,320 bytes |
| Zenodo MD5 | `cb045bfd2506d7df9316213e11c4e757` |
| Locally computed SHA-256 | `a7f318d9c64d7d2c214a5f44ee515b70c7d1ee930178b4b6b00cf5c733b0dfda` |
| Archive format | RAR v5 |
| Extraction location for this audit | `/private/tmp/tibyan_archive` only; nothing downloaded or extracted was committed. |

## Released file structure

The archive extracts under `Data/` and contains:

| Path | Role observed | Lines | SHA-256 |
|---|---:|---:|---|
| `Data/Final-Data-after-human-annotation/Tibyan Correct.txt` | Final corrected side | 6,192 | `5f8fde9319df89419ade12114f14466301cafb181ce807df896fb5de2361c4e4` |
| `Data/Final-Data-after-human-annotation/Tibyan Incorrect.txt` | Final erroneous side | 6,192 | `e3b72a45abffbf2a63da07912adc0fa0b29a41c495d609aa2b917fc4d58956a6` |
| `Data/Data-befor-human-annotation(Generated_by_Model)/correct.txt` | Generated corrected side before human annotation | 3,625 | `734b7c625fac25c6600e432d3173af4d4fb8db01b5ba5c3a92919dfd73099c49` |
| `Data/Data-befor-human-annotation(Generated_by_Model)/incorrect.txt` | Generated erroneous side before human annotation | 3,626 | `b21e8515aa5932e79645525b21d42c870146c6820e1ba531071b0f0e5412293a` |
| `Data/Guide Sentence/` | Source guide material, including the A7'ta corpus folder and many text files | 481 files | Not enumerated in Git to avoid noisy, text-bearing provenance output. |

Observed schema for the final data:

- UTF-8 with BOM handling (`utf-8-sig`).
- Plain text, no header row.
- One record per line in each final file.
- The corrected and erroneous final files have equal line counts and are paired by
  line number.
- No stable record ID, split field, error-label field, explanation field, or
  source-provenance field is present in the final paired files.
- Two final line pairs are identical between corrected and erroneous sides; a
  future manifest task should flag these explicitly.

## Statistics checked without publishing corpus text

| Check | Result |
|---|---|
| Final corrected lines | 6,192 |
| Final erroneous lines | 6,192 |
| Empty final lines | 0 in both final files |
| Identical final line pairs | 2 |
| Approximate whitespace tokens, corrected side | 596,749 |
| Approximate whitespace tokens, erroneous side | 572,572 |
| Pair-identity SHA-256 over final line-pair hashes | `5400db3e036ee54b8e38760679374ad666e6f8661184d826605b751fbbbcc795` |

The paper reports 618,598 words for correct data and 604,592 words for incorrect
data. The local counts above are simple whitespace-token counts over the released
final files, so they should be treated as an audit signal, not as a correction to
the paper. Resolve the counting definition before citing final token totals in a
paper or experiment report.

## Splits

No official train/dev/test split was found in the paper text, arXiv source, Zenodo
metadata, or extracted archive structure. Do not invent a split inside this
intake task. Any Musahhih split for Tibyan should be created in a separate
methodology-reviewed manifest task and recorded as a project split, not an
official dataset split.

## Expert-validation evidence

The paper supports a claim that the corpus was reviewed and refined with
linguistic/professional annotators. It describes two annotation phases: first,
annotators corrected morphology, punctuation, spelling, syntax, word choice, and
dialectal usage while preserving the intended wording; second, a qualified
linguistic auditor reviewed the text for accuracy and error freedom. The archive
itself does not include reviewer identities, adjudication logs, or annotation
guideline files beyond the text files, so Musahhih should cite the paper for this
claim and avoid implying independent expert validation by this project.

## Overlap checks

No corpus text was printed or committed.

| Dataset | Status |
|---|---|
| Nahw-Passage | Public Nahw data was downloaded with `scripts/download_nahw.py`; exact SHA-256 matching found zero full-string matches between Tibyan final lines and Nahw `passage`, `error`, `correction`, or `explanation` fields. |
| QALB | Not run in this branch because private QALB text is not available here. This must be rerun in the private QALB environment before training. |

## Follow-up required before modeling

1. Create a private, Git-ignored Tibyan manifest that records archive SHA-256,
   final file SHA-256 values, line-pair IDs, split assignment, duplicate flags,
   and exact-overlap flags without copying public corpus text into Git.
2. Decide a Musahhih train/dev split policy for Tibyan, because the release has
   no official split.
3. Rerun exact-overlap checks against private QALB train/dev/test data and keep
   QALB text private.
4. Investigate the token-count discrepancy between the paper and the released
   final files before citing token totals in a publication.
5. Preserve attribution and CC BY 4.0 license notices in any internal manifest,
   model card, or released derivative artifact.
