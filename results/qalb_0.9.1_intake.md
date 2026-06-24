# QALB 0.9.1 intake report

Date: 2026-06-23

## Provenance and handling

- Release: `QALB-0.9.1-Dec03-2021-SharedTasks`
- Archive size: 94,288,190 bytes
- Archive SHA-256: `c14764b01439618bdcebda04bd5b9365cd70a1fbc58607f1bd18cf357514e503`
- Archive inventory: 277 ZIP entries, including 220 files; 424,377,838 uncompressed bytes
- Safety check: no absolute paths, path traversal entries, or encrypted entries
- Local raw location: `data/raw/qalb/` (ignored by Git)

The license grants no-fee use and copying solely for internal research and evaluation. It does not grant redistribution, sublicensing, dataset modification, assignment, or commercial rights. Raw QALB data therefore remains private and untracked. The original archive, `README.txt`, and `LICENSE.txt` are retained locally without modification. Processing should use manifests that reference unchanged records; obtain institutional guidance before creating persistent transformed corpus copies.

For routine work, the original `.sent`, `.cor`, and `.m2` files were extracted byte-for-byte. Large `.column` feature files, submitted-system outputs, and archival documents were not extracted; they remain available inside the original ZIP.

## Split integrity

| Split | Records | Unique IDs | Duplicate source records | M2 annotation lines |
|---|---:|---:|---:|---:|
| QALB-2014 L1 train | 19,411 | 19,411 | 105 | 306,757 |
| QALB-2014 L1 dev | 1,017 | 1,017 | 1 | 16,659 |
| QALB-2014 L1 test | 968 | 968 | 3 | 16,378 |
| QALB-2015 L1 test | 920 | 920 | 12 | 13,299 |
| QALB-2015 L2 train | 310 | 310 | 0 | 13,206 |
| QALB-2015 L2 dev | 154 | 154 | 0 | 7,293 |
| QALB-2015 L2 test | 158 | 158 | 0 | 6,647 |

Checks performed directly against the supplied ZIP and again against the selected extraction:

- every `.sent`, `.cor`, and `.m2` file decodes as UTF-8 with BOM handling;
- source, corrected, and M2 document counts agree within every split;
- ordered `.sent` source text exactly matches ordered M2 `S` records;
- every document ID is unique within its official split;
- no exact QALB source document matches any of the 100 unique Nahw-Passage passages.

There are 22,938 records and 22,816 unique source strings across the seven distributed split/track combinations. The difference includes documented duplicates within splits and one exact cross-split duplicate.

## Leakage finding

One source string occurs in both QALB-2015 L2 train and test under different document IDs. The source and corrected text are identical, meaning it is an already-correct record. To identify it without publishing corpus text:

- source UTF-8 SHA-256: `32f52ef800b5292b2b3df1e0dfe6ba5b6254d25a32dbad12909dcd8e1f144e5b`
- source length: 34 Unicode code points
Original document identifiers are intentionally omitted from public documentation; the private selection manifest locates the excluded record.

The official raw files must remain unchanged. Before any training run, a private selection manifest must exclude the train-side record at load time and record that decision. All QALB test splits remain strictly evaluation-only.

## Decision

QALB 0.9.1 is accepted for internal research preparation subject to its license. No QALB model training or benchmark evaluation was run during this intake.

## Manifest generation

The private selection manifests were generated with:

```bash
python scripts/prepare_qalb_manifests.py
```

- Generator commit: `8ec8d2aab37d364685b0a066c73e9c0ff5111a02`
- QALB input SHA-256: `c14764b01439618bdcebda04bd5b9365cd70a1fbc58607f1bd18cf357514e503`
- Nahw input SHA-256: `97d4f5e0b75ff5848ffdff113a74676c0de607d0bb877e1f26c1bde1585a2208`
- Registry records: 22,938
- Training records selected: 19,720
- Development records selected: 1,171
- Official test records retained as evaluation-only: 2,046
- Train/dev exact QALB-test overlaps excluded: 1
- Train/dev exact Nahw overlaps excluded: 0

The generator reads the unchanged QALB ZIP directly. Exact overlap comparisons apply no normalization; they use exact UTF-8 strings after removing file-format prefixes. Within-split duplicates remain selected. The `duplicate_source_within_split` flag covers all 237 records belonging to a duplicated-source group, whereas the 121 duplicate excess records count only occurrences after the first within each group (total records minus within-split unique-source counts).

| Private output under `data/processed/qalb/` | SHA-256 |
|---|---|
| `qalb_registry.jsonl` | `e0a87eb3b6bdf9d0dca4edd29e4a4ab72b8c6a49d2e29f83aaab496147939691` |
| `qalb_train_selection.jsonl` | `9c9a054120d884a26d1b700501020452211df7b24de7e64476615d4a85d5dca2` |
| `qalb_dev_selection.jsonl` | `563b12a75789ce0865ab341614935d855ab42086fae6e0cdaa26ba17f4de26c8` |
| `qalb_manifest_summary.json` | `ee413cb049284a7115ee6c75e654ce9f7151207bc2aa3553a245e24d25931155` |

Validation passed for a deterministic rerun and for the output schema/privacy checks. Every generated file is a corpus-text-free metadata/hash artifact under an ignored, private path; none is tracked. This manifest-only run performed no model training and no QALB benchmark evaluation.
