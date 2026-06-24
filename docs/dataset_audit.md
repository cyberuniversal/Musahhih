# Dataset Audit

| Dataset | What it contains | Intended role | Current status | Main risk |
|---|---|---|---|---|
| Nahw-Passage | 100 MSA passages represented as 511 error/correction/explanation records | **Test only** for GED/GEC/GEX | Public in Nahw GitHub | Training on it would invalidate the benchmark |
| Nahw-MCQ | 5K natural Arabic grammar MCQs | GU evaluation; not primary GEC training data | Public in Nahw GitHub | Different task format from passage correction |
| Nahw Synthetic 10K | Synthetic grammar MCQs | GU replication/ablation | Public in Nahw GitHub | MCQ task mismatch for GEC |
| QALB-2014 | Parallel erroneous/corrected native-speaker Arabic text | Natural training/dev/test benchmark | Release 0.9.1 received and integrity-audited locally on 2026-06-23 | Research-only license; no redistribution or dataset modification rights; preserve official splits |
| QALB-2015 | Native-speaker test and non-native-speaker train/dev/test correction data | Additional benchmark or training split | Release 0.9.1 received and integrity-audited locally on 2026-06-23 | One exact L2 source occurs in both train and test; exclude the train-side duplicate from any derived training view |
| Tibyan | Large synthetic Arabic GEC corpus with expert validation | Synthetic or mixed-data training | Paper is public; locate the authoritative released dataset and license | Synthetic style artifacts; version provenance |
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
