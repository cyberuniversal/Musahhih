# Dataset Audit

| Dataset | What it contains | Intended role | Current status | Main risk |
|---|---|---|---|---|
| Nahw-Passage | 100 MSA passages represented as 511 error/correction/explanation records | **Test only** for GED/GEC/GEX | Public in Nahw GitHub | Training on it would invalidate the benchmark |
| Nahw-MCQ | 5K natural Arabic grammar MCQs | GU evaluation; not primary GEC training data | Public in Nahw GitHub | Different task format from passage correction |
| Nahw Synthetic 10K | Synthetic grammar MCQs | GU replication/ablation | Public in Nahw GitHub | MCQ task mismatch for GEC |
| QALB-2014 | Parallel erroneous/corrected native-speaker Arabic text | Natural training/dev/test benchmark | Release 0.9.1 received and integrity-audited locally on 2026-06-23 | Research-only license; no redistribution or dataset modification rights; preserve official splits |
| QALB-2015 | Native-speaker test and non-native-speaker train/dev/test correction data | Additional benchmark or training split | Release 0.9.1 received and integrity-audited locally on 2026-06-23 | One exact L2 source occurs in both train and test; exclude the train-side duplicate from any derived training view |
| Tibyan | Large synthetic Arabic GEC corpus with paper-described expert/professional review | Conditional synthetic or mixed-data training source after private manifesting and overlap checks | Authoritative Zenodo release located and verified on 2026-06-25: `Tibyan-corpus`, DOI `10.5281/zenodo.14623621`, CC BY 4.0 | No official splits or IDs; paper/release token-count discrepancy; QALB overlap still requires private check |
| ZAEBUC-related data | Learner writing used by recent GED/GEC work | Optional cross-domain evaluation | Check paper/repository access terms | Domain differs from Nahw passages |
| ARETA | Automatic error-type annotation tool | Error analysis only | Public research tool/paper | Automatic tags are imperfect |

## Rules

1. Record the exact URL, license, version, checksum, and retrieval date for every dataset.
2. Preserve original train/dev/test splits.
3. Deduplicate across datasets before training.
4. Never report a benchmark that overlaps with training data.
5. Keep a manifest of every transformation.

## QALB 0.9.1 intake

- Source: the registered QALB shared-task distribution supplied to the team.
- Release: `QALB-0.9.1-Dec03-2021-SharedTasks`, dated 2021-12-03.
- Archive SHA-256: `c14764b01439618bdcebda04bd5b9365cd70a1fbc58607f1bd18cf357514e503`.
- Retrieved by the project: 2026-06-23.
- License: internal research and evaluation use only. The release prohibits sublicensing, redistribution, dataset modification, and assignment of the license. Copyright and license notices must accompany internal copies. Commercial use or additional rights require permission from the rights holders.
- Repository handling: the archive and extracted corpus files remain under ignored `data/raw/qalb/`. They must not be committed, attached to a public release, or copied into public experiment artifacts. Use manifests that reference unchanged originals; obtain institutional guidance before creating persistent transformed corpus copies, and never publish them without additional permission.

The release contains these official document counts:

| Track | Train | Dev | Test |
|---|---:|---:|---:|
| QALB-2014 L1 (native) | 19,411 | 1,017 | 968 |
| QALB-2015 L1 (native) | Reuses the 2014 native data | Reuses the 2014 native data | 920 |
| QALB-2015 L2 (non-native) | 310 | 154 | 158 |

All `.sent`, `.cor`, and `.m2` files decoded as UTF-8 with BOM handling. Within every split, document IDs are unique, source/reference record counts agree, and `.sent` source text matches the ordered `S` records in `.m2`.

Exact-source checks found no overlap between any QALB source document and the 100 unique Nahw-Passage passages. This is only an exact-text check and does not rule out paraphrase or broader provenance overlap. One 34-character, already-correct L2 source occurs in both QALB-2015 L2 train and test under different document IDs. Its SHA-256 is `32f52ef800b5292b2b3df1e0dfe6ba5b6254d25a32dbad12909dcd8e1f144e5b`. Preserve the official raw files, but exclude the train-side occurrence at load time through a private selection manifest before modeling.

### Private selection manifests

The verified manifest run used `python scripts/prepare_qalb_manifests.py` from generator commit `303bd4774d7a4fbcf7c346c253b6429255c58fff`. The generator read the unchanged QALB ZIP directly and compared exact source strings without normalization; file-format prefixes were removed before the exact UTF-8 comparisons.

| Input | SHA-256 |
|---|---|
| QALB 0.9.1 ZIP | `c14764b01439618bdcebda04bd5b9365cd70a1fbc58607f1bd18cf357514e503` |
| Nahw-Passage JSON | `97d4f5e0b75ff5848ffdff113a74676c0de607d0bb877e1f26c1bde1585a2208` |

| Private output under `data/processed/qalb/` | SHA-256 |
|---|---|
| `qalb_registry.jsonl` | `e0a87eb3b6bdf9d0dca4edd29e4a4ab72b8c6a49d2e29f83aaab496147939691` |
| `qalb_train_selection.jsonl` | `9c9a054120d884a26d1b700501020452211df7b24de7e64476615d4a85d5dca2` |
| `qalb_dev_selection.jsonl` | `563b12a75789ce0865ab341614935d855ab42086fae6e0cdaa26ba17f4de26c8` |
| `qalb_manifest_summary.json` | `e322424d4b1e0265c8c3011a243f7aba38609c35e3905b4aedf0b0ef75e3ea33` |

The registry contains 22,938 records. The selection contains 19,720 training records and 1,171 development records, while all 2,046 official test records remain evaluation-only. One train/dev record with exact QALB-test source overlap was excluded; no train/dev record was excluded for exact Nahw overlap.

Within-split duplicated sources are intentionally preserved. The manifest flags 237 records because every member of a duplicated-source group receives the flag. This is distinct from the 121 duplicate excess records, which count only records beyond the first occurrence in each within-split group (total records minus within-split unique-source counts).

The deterministic rerun and schema/privacy checks passed. All outputs contain no source, correction, annotation, prompt, or passage text; they remain ignored and private and must not be committed or redistributed. Obtain institutional guidance before creating any persistent transformed corpus copy. This preparation performed no model training and no QALB test evaluation.

See `results/qalb_0.9.1_intake.md` for the reproducible intake checks and duplicate counts.

## Tibyan intake

- Source paper: Ahlam Alrehili and Areej Alhothali, “Tibyan Corpus: Balanced and
  Comprehensive Error Coverage Corpus Using ChatGPT for Arabic Grammatical Error
  Correction,” `arXiv:2411.04588`, DOI `10.48550/arXiv.2411.04588`.
- Authoritative dataset release: Zenodo record `14623621`,
  <https://zenodo.org/records/14623621>, DOI `10.5281/zenodo.14623621`.
- Retrieved and availability-tested: 2026-06-25.
- Zenodo creator metadata: `Alrehili, Ahlam`, King Abdulaziz University, ORCID
  `0000-0002-4218-4659`.
- License: CC BY 4.0 (`cc-by-4.0`). Attribute the creator, link the license,
  indicate changes, and do not add legal or technical restrictions beyond the
  license. Other rights may still limit specific uses.
- Archive: `Data (1).rar`, 5,614,320 bytes, Zenodo MD5
  `cb045bfd2506d7df9316213e11c4e757`, locally computed SHA-256
  `a7f318d9c64d7d2c214a5f44ee515b70c7d1ee930178b4b6b00cf5c733b0dfda`.
- Final released pairs:

| File | Lines | SHA-256 |
|---|---:|---|
| `Data/Final-Data-after-human-annotation/Tibyan Correct.txt` | 6,192 | `5f8fde9319df89419ade12114f14466301cafb181ce807df896fb5de2361c4e4` |
| `Data/Final-Data-after-human-annotation/Tibyan Incorrect.txt` | 6,192 | `e3b72a45abffbf2a63da07912adc0fa0b29a41c495d609aa2b917fc4d58956a6` |

The final files are UTF-8 text with BOM handling, one line per paired record, and
no header, stable ID, official split field, error-label field, explanation field,
or per-record provenance field. The two final files have equal line counts; two
line pairs are identical between corrected and erroneous sides. A simple
whitespace-token count over the final files produced 596,749 corrected-side tokens
and 572,572 erroneous-side tokens, which does not exactly match the paper's
reported 618,598 and 604,592 word counts. Treat this as an unresolved counting
definition or release-version issue until the private manifest task resolves it.

No official train/dev/test split was found in the paper, arXiv source package,
Zenodo metadata, or extracted archive. Any Musahhih split must be created as a
separate methodology-reviewed project split, not represented as an official
Tibyan split.

The paper supports a claim of linguistic/professional review, including two
annotation phases and a qualified linguistic auditor, but the archive does not
include reviewer identities or adjudication logs. Cite the paper for that claim
and do not claim that Musahhih independently validated the Arabic.

Hash-only overlap checks against public Nahw-Passage found zero exact full-string
matches between Tibyan final lines and Nahw `passage`, `error`, `correction`, or
`explanation` fields. QALB overlap was not run in this branch because private
QALB text is not available here; it must be checked in the private QALB
environment before any Tibyan training or mixed-data experiment.

Recommendation: conditionally usable. Create a private, Git-ignored Tibyan
manifest before modeling, preserve CC BY 4.0 attribution/license notices, define
a project split without touching Nahw-Passage, rerun exact QALB overlap checks,
and resolve the token-count discrepancy before publication.

See `results/tibyan_dataset_intake.md` for the corpus-text-free intake details.
