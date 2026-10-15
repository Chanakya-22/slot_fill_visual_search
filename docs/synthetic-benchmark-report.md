# Synthetic Visual-Search Benchmark Report

## Scope and headline

This report records the completed seed-42 runs for the three retained models
on the project's **synthetic-only** visual-search task. The primary experiment
holds out products; the secondary experiment holds out query templates. All
three models used the same split and labels within each experiment. No model
was selected or tuned using test metrics.

On the product-held-out split, all three models scored 1.000 on intent
accuracy, intent macro F1, and exact entity-level slot precision, recall, and
F1. Replacing images with zero-valued normalized pixels did not change the two
multimodal models' reported product-test scores. These results expose an
important task limitation: text templates and slot-bearing words appear
sufficient for this generated task. They are **not evidence of visual
contribution or real-world generalization**.

## Dataset

- 1,000 distinct query examples over 100 synthetic products and 24 query
  templates; no duplicate rows were used to inflate the count.
- Five balanced intent classes (200 examples each): `search_product`,
  `troubleshoot_product`, `identify_product`, `check_compatibility`, and
  `request_specifications`.
- Images were rendered locally with Pillow using
  `procedural_pillow_vector_v2`. Queries, image appearance, and structured
  catalogue metadata were generated synthetically with seed 42.
- The result files record dataset/catalogue hashes, label mappings, split IDs,
  class distributions, model metrics, and per-example test predictions.
- The validator passed checks for image files, labels, token/tag alignment,
  diversity, uniqueness, and fixed-seed reproducibility. It also reported 100
  unreferenced legacy image filenames whose numeric IDs overlap canonical
  products; these were left untouched and are not referenced by the active
  CSV.

## Models and protocol

The comparison retains the existing `BaselineTextModel`,
`BaselineConcatModel`, and `CrossAttentionModel` architectures. All use
`bert-base-uncased`; multimodal models use `openai/clip-vit-base-patch32`.
The text baseline has no vision encoder. The concatenation model trains the
BERT and CLIP vision parameters. Cross-attention trains BERT and its
cross-attention layers plus the final two CLIP vision layers; earlier CLIP
vision layers are frozen. The proposed cross-attention model receives the
image tensor in its forward pass.

Environment: Python 3.12.10, PyTorch 2.13.0+cpu, Transformers 5.15.0, CPU
execution with 8 PyTorch threads; CUDA was unavailable. Training used batch
size 8, learning rate `3e-5`, slot-loss weight 1.0, up to 5 epochs, and
early-stopping patience 2. Checkpoint selection used validation
entity-level slot F1. The same partition sample IDs and labels were used by
all three architectures in each run.

### Splits

| Experiment | Split unit | Train | Validation | Test | Group counts |
|---|---|---:|---:|---:|---|
| Primary | Product | 700 (70%) | 150 (15%) | 150 (15%) | 70 / 15 / 15 products |
| Secondary | Query template | 501 (50.1%) | 248 (24.8%) | 251 (25.1%) | 12 / 6 / 6 templates |

Product groups are disjoint in the primary experiment, and each intent has
140/30/30 train/validation/test examples. Query templates can overlap across
those product partitions. Template groups are disjoint in the secondary
experiment, with all intents represented in every partition; products can
overlap there. Thus the secondary run tests wording generalization, not
unseen-product generalization. One seed was run for each split.

### Metrics and measurement

- Intent accuracy is the fraction of examples with the correct intent.
- Intent macro F1 is the unweighted mean of per-class intent F1. Class-wise
  precision, recall, F1, and support are reported separately.
- Slot precision, recall, and F1 are micro-aggregated exact entity-span-and-
  type metrics from BIO sequences; a partial span or wrong entity type is not
  a match. Special tokens, padding, and ignored WordPieces are excluded.
- The selected checkpoint is the best validation entity-level slot-F1
  checkpoint; intent metrics are tracked independently, not used to select
  the checkpoint.
- Training time is the recorded training/validation/checkpointing duration;
  it excludes model initialization, final test evaluation, and latency
  measurement. Batch-size-one latency is the mean of 5 timed model
  forward/device-transfer runs after 2 warmups, excluding preprocessing.

## Results

### Primary: product-held-out test set

All test partitions contain 30 examples per intent. Every model produced
precision, recall, and F1 of 1.000 for every intent class (support 30 each).
Intent confusion matrices are diagonal.

| Model | Best epoch | Intent accuracy | Intent macro F1 | Slot precision | Slot recall | Slot F1 | Train time (s) | Latency (ms) | Total / trainable parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BaselineTextModel | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 329.08 | 36.66 | 109,494,544 / 109,494,544 |
| BaselineConcatModel | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 782.69 | 75.19 | 196,962,832 / 196,962,832 |
| CrossAttentionModel (proposed) | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 351.25 | 74.63 | 198,723,856 / 125,443,600 |

There were no incorrect intent predictions or slot-tag mismatches among the
150 product-test examples for any model. Both multimodal models also scored
1.000 intent accuracy, macro F1, and slot F1 when the normalized image inputs
were replaced with zeros. This is a warning that the primary test set does
not demonstrate reliance on visual information.

### Secondary: query-template-held-out test set

The test set contains 251 examples: 50 for each intent except
`request_specifications`, which has 51.

| Model | Best epoch | Intent accuracy | Intent macro F1 | Slot precision | Slot recall | Slot F1 | Train time (s) | Latency (ms) | Total / trainable parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BaselineTextModel | 2 | 0.801 | 0.726 | 0.941 | 0.941 | 0.941 | 192.65 | 36.85 | 109,494,544 / 109,494,544 |
| BaselineConcatModel | 2 | 0.793 | 0.714 | 0.933 | 0.941 | 0.937 | 546.72 | 89.79 | 196,962,832 / 196,962,832 |
| CrossAttentionModel (proposed) | 1 | 0.857 | 0.837 | 0.925 | 0.945 | 0.935 | 227.20 | 66.68 | 198,723,856 / 125,443,600 |

Class-wise intent precision / recall / F1:

| Intent | BaselineTextModel | BaselineConcatModel | CrossAttentionModel |
|---|---|---|---|
| `search_product` | 0.862 / 1.000 / 0.926 | 1.000 / 0.960 / 0.980 | 1.000 / 0.340 / 0.507 |
| `troubleshoot_product` | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| `identify_product` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 1.000 / 0.940 / 0.969 |
| `check_compatibility` | 0.543 / 1.000 / 0.704 | 0.667 / 1.000 / 0.800 | 0.833 / 1.000 / 0.909 |
| `request_specifications` | 1.000 / 1.000 / 1.000 | 0.654 / 1.000 / 0.791 | 0.662 / 1.000 / 0.797 |

The secondary split is materially harder and the models exhibit different
class confusions. For example, both baselines have zero F1 on
`identify_product`; cross-attention improves that class but has only 0.340
recall on `search_product`. The single-seed outcomes do not establish that
one architecture is generally better.

Zero-image ablation on the template-held-out test set:

| Model | Input | Intent accuracy | Intent macro F1 | Slot F1 |
|---|---|---:|---:|---:|
| BaselineConcatModel | Paired image | 0.7928 | 0.7141 | 0.9371 |
| BaselineConcatModel | Zero image | 0.7928 | 0.7141 | 0.9365 |
| CrossAttentionModel | Paired image | 0.8566 | 0.8365 | 0.9351 |
| CrossAttentionModel | Zero image | 0.8526 | 0.8269 | 0.9260 |

Image removal changes some scores slightly in the secondary run, but this
ablation is not sufficient to claim meaningful visual grounding. Product
metadata and intent/slot wording are procedurally linked, the sample is
small, and the result is from one seed.

## Error analysis and examples

The product-held-out run had zero measured errors, so there are no incorrect
product-test examples to report. On the template-held-out test set, the
error-analysis artifacts record 50 / 52 / 36 intent errors and 50 / 58 / 65
slot-token mismatches for BaselineTextModel / BaselineConcatModel /
CrossAttentionModel, respectively. These are counts of test examples with
wrong intent and token-level slot-tag mismatches, respectively; they are not
entity-level F1 values.

Examples from the template-held-out test set:

- Correct BaselineTextModel prediction: “search for a red gear assembly with
  a fine tooth design” → `search_product`; all listed slot tags matched.
- Incorrect CrossAttentionModel intent prediction: the same query had true
  intent `search_product` and predicted intent `request_specifications`, while
  its slot tags matched. This illustrates independent intent and slot
  behavior.
- A BaselineTextModel error was “which gear is pictured with a forest green
  finish and beveled tooth body”: true intent `identify_product`, predicted
  `check_compatibility`; its slot sequence also overextended the shape entity
  to “body” beyond the gold span.

Full representative correct/incorrect examples and transition counts are
available in the per-split error-analysis JSON files below. Confusion matrices
and learning curves are preserved as figures.

## Reproducibility artifacts and commands

Run the checks and regenerate the experiments/plots with:

```powershell
.\venv\Scripts\python.exe validate_synthetic_dataset.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe train.py --models all --split-mode product --seed 42 --epochs 5 --patience 2
.\venv\Scripts\python.exe benchmark_results.py data\processed\experiment_results_product_seed42.json
.\venv\Scripts\python.exe train.py --models all --split-mode template --seed 42 --epochs 5 --patience 2
.\venv\Scripts\python.exe benchmark_results.py data\processed\experiment_results_template_seed42.json
```

Both completed machine-readable run records and report artifacts are checked
in under [`benchmark-artifacts`](./benchmark-artifacts/):

- [Product-split raw results](./benchmark-artifacts/product_seed42.json) and
  [comparison CSV](./benchmark-artifacts/product_seed42_model_comparison.csv),
  [error analysis](./benchmark-artifacts/product_seed42_error_analysis.json),
  [comparison plot](./benchmark-artifacts/product_seed42_model_comparison.png),
  [learning curves](./benchmark-artifacts/product_seed42_learning_curves.png).
- [Template-split raw results](./benchmark-artifacts/template_seed42.json) and
  [comparison CSV](./benchmark-artifacts/template_seed42_model_comparison.csv),
  [error analysis](./benchmark-artifacts/template_seed42_error_analysis.json),
  [comparison plot](./benchmark-artifacts/template_seed42_model_comparison.png),
  [learning curves](./benchmark-artifacts/template_seed42_learning_curves.png).
- Each split also includes an intent confusion-matrix plot for every model.

Checkpoints are deliberately not included in the report artifacts; training
commands write them to the ignored `data/processed/` directory.

## Limitations and next steps

The benchmark uses only generated product images and generated queries. It
does not measure robustness to real product photography, natural user
language, unseen product domains, or real-world visual-search conditions.
The perfect primary scores and unchanged zero-image ablation indicate that
the synthetic task may be solvable from textual cues alone. The template
split reuses products, while the product split reuses query templates.
Furthermore, results use one seed, a small synthetic catalogue, and a
validation checkpoint criterion focused on slot F1. These results should be
presented as pipeline and controlled-method evidence only, not as a claim of
real-world performance or superiority.

Future work should create controlled counterfactual pairs where text is held
constant while visual evidence is necessary to distinguish products; add
independently held-out products and wording; audit label/image correlations;
and evaluate with multiple seeds plus a separately governed real-data study
if permitted.
