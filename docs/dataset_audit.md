# Dataset Audit

| Dataset | What it contains | Intended role | Current status | Main risk |
|---|---|---|---|---|
| Nahw-Passage | 100 MSA passages represented as 511 error/correction/explanation records | **Test only** for GED/GEC/GEX | Public in Nahw GitHub | Training on it would invalidate the benchmark |
| Nahw-MCQ | 5K natural Arabic grammar MCQs | GU evaluation; not primary GEC training data | Public in Nahw GitHub | Different task format from passage correction |
| Nahw Synthetic 10K | Synthetic grammar MCQs | GU replication/ablation | Public in Nahw GitHub | MCQ task mismatch for GEC |
| QALB-2014 | Parallel erroneous/corrected Arabic text | Natural training/dev/test benchmark | Access and license must be confirmed before use | Mostly orthographic edits; fixed official splits required |
| QALB-2015 | Native/non-native correction data | Additional benchmark or training split | Access and license must be confirmed before use | Do not mix official test data into training |
| Tibyan | Large synthetic Arabic GEC corpus with expert validation | Synthetic or mixed-data training | Paper is public; locate the authoritative released dataset and license | Synthetic style artifacts; version provenance |
| ZAEBUC-related data | Learner writing used by recent GED/GEC work | Optional cross-domain evaluation | Check paper/repository access terms | Domain differs from Nahw passages |
| ARETA | Automatic error-type annotation tool | Error analysis only | Public research tool/paper | Automatic tags are imperfect |

## Rules

1. Record the exact URL, license, version, checksum, and retrieval date for every dataset.
2. Preserve original train/dev/test splits.
3. Deduplicate across datasets before training.
4. Never report a benchmark that overlaps with training data.
5. Keep a manifest of every transformation.
