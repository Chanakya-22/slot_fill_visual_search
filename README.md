# Multi-Modal Intent Classification and Slot Filling for Visual Search

This project studies joint intent classification and slot filling for visual
search with pretrained BERT and CLIP encoders and three retained architectures:
text-only BERT, BERT/CLIP concatenation, and BERT/CLIP cross-attention.

## Synthetic data only

The experiment uses no external datasets or real-world product images. Generate
the local catalogue and 224 x 224 RGB images with:

```powershell
.\venv\Scripts\python.exe prepare_hardware_data.py
```

Generation is deterministic by default (`seed=42`) and produces 100 distinct
procedural products with ten differently templated queries each (1,000
examples). To regenerate with another seed or a smaller divisible sample:

```powershell
.\venv\Scripts\python.exe prepare_hardware_data.py --seed 42 --num-samples 1000
```

The generator writes `data/raw/dataset.csv`, `data/raw/catalogue.json`,
`data/raw/dataset_summary.json`, and one PNG per product under
`data/raw/images/`. Canonical IDs use four digits throughout, for example
`product_0020` and `images/product_0020.png`. Product images are drawn locally
with category-specific Pillow vector shapes, recorded color finishes,
highlights, and hardware details. Material, price, and declared compatibility
are structured catalogue metadata; they are not inferred from image pixels.
Image paths in the CSV are relative to its directory so the generated dataset
remains portable.

The image directory may also contain old, unreferenced files from prior
generators. Validation reports legacy filenames whose integer IDs overlap the
current catalogue; it does not treat them as product duplicates, overwrite,
or delete them. Only paths listed in the current CSV are part of this dataset.

## Data contract and labels

The CSV keeps the loader's original `image_path`, `text_query`, `intent_id`,
and `slot_ids` columns and adds `sample_id`, `product_id`, `intent`, `tokens`,
`slot_tags`, `product_metadata`, and `generation_info`. Each query's token list
is whitespace-tokenized and exactly reconstructs `text_query`; `slot_tags`
contains one BIO tag per query token. The loader tokenizes this explicit word
list with the fast BERT tokenizer and `word_ids()`, marks special/padding
positions `-100`, and converts additional WordPieces inside a BIO entity to
the corresponding `I-` label. Rows with a `product_id` must reference exactly
`images/{product_id}.png`, so unreferenced legacy image aliases cannot enter
the active multimodal dataset through a mismatched CSV path.

Intent IDs are defined in `src/labels.py`:

| ID | Intent |
|---:|---|
| 0 | `search_product` |
| 1 | `troubleshoot_product` |
| 2 | `identify_product` |
| 3 | `check_compatibility` |
| 4 | `request_specifications` |

The slot IDs in the same module map to `O`, `B-/I-CATEGORY`,
`B-/I-COLOR`, `B-/I-SHAPE`, `B-/I-MATERIAL`, and `B-/I-PRICE`.
The four-column CSV layout remains supported: legacy intent IDs retain their
existing purchase, troubleshoot, identify, compatibility, and specification
meanings, and legacy `slot_ids` values map to `O`, `B-CATEGORY`, and
`B-MATERIAL`. Regenerated rows include explicit `slot_tags`; new `slot_ids`
therefore use the complete BIO label map, and model slot heads now use its
full size. Price requests are represented as specification requests and carry
an explicit `PRICE` slot.

## Validation and experiment safeguards

Run dataset checks and focused tests:

```powershell
.\venv\Scripts\python.exe validate_synthetic_dataset.py
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The validator checks schema and label validity, existing images, image
dimensions, token/tag alignment, intent/template/product diversity, exact
record uniqueness, and fixed-seed reproducibility. Training requires product
IDs and uses seeded product-group splits, preventing the same product image
from appearing on both sides of the split. Intent metrics include
accuracy, macro F1, and per-class precision, recall, and F1. Slot precision,
recall, and F1 are exact-span-and-type micro metrics over BIO entities; special
tokens, padding, and ignored WordPieces do not contribute to these scores.

Run one architecture as a shorter pipeline check, then run all three models
with the shared product-group splits when ready:

```powershell
.\venv\Scripts\python.exe train.py --models BaselineTextModel --split-mode product --seed 42
.\venv\Scripts\python.exe train.py --models all --split-mode product --seed 42
```

The primary run writes an experiment JSON containing the exact shared split,
training configuration, per-epoch train/validation history, validation-selected
checkpoint, test metrics, intent confusion matrices, test predictions,
checkpoint paths, parameter counts, measured training time, and model-forward
batch-size-one latency (preprocessing excluded). Multimodal models also report
a zero-normalized-image test ablation; it is not applicable to the text-only
model. Run the report generator after training to create a comparison CSV,
metric and learning-curve plots, confusion-matrix plots, and an error-analysis
JSON:

```powershell
.\venv\Scripts\python.exe benchmark_results.py data\processed\experiment_results_product_seed42.json
```

The experiment JSON is written with `pending`/`in_progress` states before
training starts. If execution is interrupted, unfinished models remain
explicitly marked; plots and comparison rows include completed models only.

To evaluate query-template generalization instead, use
`--split-mode template`; that mode reserves disjoint template groups and uses
50/25/25 train/validation/test proportions so every intent is represented in
each partition. Product generalization uses disjoint product groups and
70/15/15 proportions. Each experiment writes best-validation checkpoints and
actual validation/test metrics to `data/processed/`, including the split IDs,
intent class distributions, measured fine-tuning time, and measured
batch-size-one inference latency. Test metrics are evaluated only after the
validation-selected checkpoint is restored.

These generated examples are suitable for pipeline and controlled-method
checks only. Their labels, images, wording, attributes, and compatibility are
procedurally linked, so scores on this synthetic set are not evidence of
real-world product-search performance. The generator and validator do not
train models or download pretrained weights.

`benchmark_sota.py` is explicitly compute-only: it uses random tensor inputs
and a randomly initialized ViLT model. Its output is not a task-quality
comparison or evidence of memory savings.
