# Literature Matrix

| Work | Main contribution | Data | Training / method | Evaluation | Limitation relevant to us |
|---|---|---|---|---|---|
| Nahw (2026) | Four-part Arabic grammar benchmark: GU, GED, GEC, GEX | 5K natural MCQs; 100 passages with 511 errors; 10K synthetic MCQs | Fine-tunes Gemma-3-4B-it only for GU | Accuracy, F1, correction accuracy, expert GEX ratings | Improvement experiment centers on MCQ grammar understanding, not a full fine-tuned GEC system |
| [Beyond English (2023)](https://aclanthology.org/2023.arabicnlp-1.9/) | Compares prompted and instruction-fine-tuned LLMs with seq2seq/seq2edit Arabic GEC systems; studies ChatGPT corruption and reverse-noising augmentation | QALB-2014 L1 train/dev/test; a mixed/ambiguous QALB-2015 setup whose prose excludes document-level L2 test but whose Table 1 labels the 2015 row L2 | 1/3/5-shot CoT and expert prompting; translated-Alpaca then QALB instruction tuning; AraT5v2 and GECToR-style baselines; 10K to 11M synthetic examples | Official shared-task M2 precision, recall, F1, and F0.5; normalization and ARETA diagnostics | Same-test comparisons support prompt and synthetic baselines, but model families, training data, and API coverage differ; headline results and the 2015 split description conflict with tables; no accompanying code/data release was located |
| Advancements in Arabic GED/GEC (2023) | Strong Arabic GED/GEC models and multitask experiments | QALB and ZAEBUC-related data | Arabic pretrained models, morphology, multitask setups | Shared-task metrics | Does not answer Nahw's natural-versus-synthetic extension using a modern open instruction model |
| QALB 2014/2015 | Standardized Arabic text-correction tasks | Native and non-native Arabic writing | Shared task, many systems | Official shared-task scorer | Error distribution is heavily orthographic; grammar is not the dominant category |
| Tibyan | Large, balanced synthetic Arabic GEC corpus validated by linguists | Approximately 600K tokens; multiple error families | ChatGPT-assisted generation plus expert validation | Corpus analysis with ARETA | Synthetic artifacts and access/version details must be verified before use |
| ARETA | Automatic Arabic error-type annotation | Arabic learner/correction corpora | Rule/morphology-based automatic annotation | Error-type F1 | Useful for analysis, but automatic labels are not equivalent to expert gold labels |

## Candidate gap

Nahw demonstrates that open models are weak on practical Arabic grammar and shows that fine-tuning helps GU. It explicitly leaves GED, GEC, and GEX fine-tuning as future work. A focused extension is to fine-tune an open model for **GEC**, compare natural, synthetic, and mixed training data, test on held-out Nahw-Passage and standard GEC benchmarks, and measure whether general Arabic capabilities are retained.

## Beyond English evidence notes

Primary source: Sang Kwon, Gagan Bhatia, El Moatez Billah Nagoudi, and
Muhammad Abdul-Mageed. 2023. “Beyond English: Evaluating LLMs for Arabic
Grammatical Error Correction.” *Proceedings of ArabicNLP 2023*, pages 101–119.
Association for Computational Linguistics. DOI
[`10.18653/v1/2023.arabicnlp-1.9`](https://doi.org/10.18653/v1/2023.arabicnlp-1.9).

### Experimental design

- The paper studies Arabic sentence-level GEC using QALB-2014 and QALB-2015.
  It compares ChatGPT-3.5 Turbo and GPT-4 prompting; instruction tuning of
  LLaMA-7B, Vicuna-13B, Bactrian-X-BLOOM-7B, and Bactrian-X-LLaMA-7B;
  seq2seq models mT0, mT5, AraBART, and AraT5v2; and GECToR-style ARBERTv2
  and MARBERTv2 sequence editors.
- The QALB-2014 row in Table 1 is L1 and reports 19,411 train, 1,017
  development, and 968 test sentences. The QALB-2015 row reports 310 train,
  154 development, and 920 test sentences and labels all three counts L2.
  However, the accompanying prose says that the study uses the 2015 L1 test
  set and excludes the document-level L2 test set. The paper therefore does
  not provide an internally consistent, exact description of the 2015 split
  composition; do not silently resolve this discrepancy.
- Few-shot CoT and expert prompts use one, three, or five labeled source/target
  examples from the original development set. The authors report that initial
  zero-shot and Arabic-language prompt outputs included extra explanation and
  could not be evaluated automatically without substantial preprocessing.
- Decoder-only LLMs are instruction-tuned for four epochs after first training
  on an NLLB-translated Alpaca dataset and then on QALB. The main text reports
  learning rate `2e-5` and batch size 4; Appendix Table 14 reports train batch
  size 8, gradient accumulation 8, and total batch size 64. Seq2seq models use
  50 epochs, early-stopping patience 5, learning rate `5e-5`, and total batch
  size 32. Seq2edit models use up to 100 epochs per stage, patience 5, learning
  rate `1e-5`, and total batch size 64.
- Non-API models use three runs with seeds 22, 32, and 42, development-set
  selection, and blind test evaluation. The paper reports means and standard
  deviations for these systems; API prompting rows are single reported values.
- Synthetic experiments include 10,000 QALB-2014 train sentences corrupted by
  ChatGPT using an Arabic Learner Corpus taxonomy (`syntheticGPT`), reverse
  models trained on gold or synthetic pairs (`reverseGold`, `reverseGPT`),
  scaled 5M/10M reverseGold sets, and the 11M-example Arabic GEC corpus from
  Solyman et al. (2021).

### Metrics and selected results

The official shared-task MaxMatch (M2) scripts score edit precision, recall,
F1, and precision-weighted F0.5. The tables' “Exact Match” heading denotes the
unnormalized text condition, not sentence exact-match accuracy. These metrics
are not directly comparable to Nahw-Passage highlighted-token correction
accuracy.

| Setting | Evaluation | Precision | Recall | F1 | F0.5 |
|---|---|---:|---:|---:|---:|
| ChatGPT-3.5, 5-shot expert prompt | QALB-2014 test, M2 | 66.53 | 61.62 | 63.98 | 65.49 |
| GPT-4, 5-shot CoT | QALB-2014 test, M2 | 69.46 | 61.96 | 65.49 | 67.82 |
| AraT5v2 baseline | QALB-2014 test, M2 | 73.04 ± 0.10 | 63.09 ± 0.15 | 67.70 ± 0.12 | 70.81 ± 0.11 |
| AraT5v2, 11M synthetic corpus, top-p | QALB-2014 test, M2 | 76.94 ± 0.67 | 69.26 ± 0.73 | 72.90 ± 0.68 | 75.27 ± 0.67 |
| AraT5v2, 11M synthetic corpus, top-p | QALB-2015 test, M2 | 72.64 ± 0.32 | 74.21 ± 0.75 | 73.41 ± 0.51 | 72.94 ± 0.39 |

The paper does not provide a fully controlled causal comparison of prompting
and fine-tuning: systems differ in architecture, size, pretraining, task
training data, decoding, and number of runs, and expensive API coverage is
incomplete. It does provide a common QALB-2014 test/scorer comparison.

### Reproducibility gaps and internal inconsistencies

- The abstract calls GPT-4's 65.49 F1 result “expert prompting,” while Table 2
  and the surrounding text place the GPT-4 5-shot row under CoT. The separate
  ChatGPT-3.5 5-shot expert-prompt row is 63.98 F1.
- The abstract reports best F1 values of 73.29 on QALB-2014 and 73.26 on
  QALB-2015. Those values do not appear in the result tables. Table 3 reports
  72.90 and 73.41 for the 11M/top-p system; Table 8 reports 72.90 and 72.84 in
  its unnormalized comparison.
- Table 8 labels a QALB-2015 ChatGPT row “3-shot + EP” with 49.83 F1, whereas
  Appendix Table 11 labels the 49.83 F1 row 5-shot CoT. Treat the 2015 prompt
  configuration as unresolved.
- The paper states that all datasets are publicly available, but neither the
  paper nor its ACL Anthology record links an accompanying repository,
  generated synthetic datasets, trained checkpoints, or run artifacts. No
  authoritative release for those study-specific artifacts was located during
  this review.
- The authors identify MSA-only coverage, unknown ChatGPT Arabic pretraining
  data, and sentence-level evaluation imposed by the then-current 4,097-token
  API limit as limitations. Unknown API training data also leaves a benchmark
  contamination risk that the paper cannot exclude.

### Implications for Musahhih

1. Keep B1 few-shot and B2 expert-style prompting as frozen, separate baseline
   families. The paper motivates both, but its prompt-label inconsistencies are
   a reason not to copy an implied “best” template after viewing test results.
2. Continue using eligible QALB train records for B1 demonstrations and QALB
   development only for technical validation. Do not imitate the paper's use
   of development examples if that would alter Musahhih's frozen protocol.
3. Validate response parsing and artifact capture before final evaluation;
   the paper's zero-shot and Arabic-prompt outputs were not automatically
   scorable without additional processing.
4. Report QALB M2 P/R/F1/F0.5 separately from Nahw-Passage token correction
   accuracy. A score from one benchmark cannot serve as the other's baseline.
5. Preserve Musahhih's matched natural/synthetic/mixed comparisons. The paper
   supports a synthetic-data quality hypothesis, but its 5M, 10M, and 11M
   sources differ, so size and source quality are not isolated cleanly.
