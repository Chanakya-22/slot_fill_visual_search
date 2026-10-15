"""Validate the generated synthetic visual-search catalogue without model downloads."""

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image

from src.labels import INTENT_NAME_TO_ID, SLOT_TAG_TO_ID
from src.synthetic_data import (
    QUERY_TEMPLATES,
    RENDERER_NAME,
    generate_synthetic_dataset,
)


def _read_json(value, field, row_number):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid JSON in {field} at CSV row {row_number}.") from exc


def validate_dataset(
    csv_path: Path | str, min_examples: int = 600, min_products: int = 25
) -> dict:
    import csv

    csv_path = Path(csv_path).resolve()
    summary_path = csv_path.parent / "dataset_summary.json"
    required_columns = {
        "image_path",
        "text_query",
        "intent_id",
        "slot_ids",
        "sample_id",
        "product_id",
        "intent",
        "tokens",
        "slot_tags",
        "product_metadata",
        "generation_info",
    }
    with csv_path.open(newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        missing = required_columns.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        records = list(reader)

    if len(records) < min_examples:
        raise ValueError(f"Expected at least {min_examples} examples; found {len(records)}.")
    if not records:
        raise ValueError("Dataset contains no examples.")

    intent_counts = Counter()
    slot_counts = Counter()
    template_ids = set()
    products = {}
    product_examples = Counter()
    record_keys = set()
    image_hashes = set()
    colors = set()
    shapes = set()

    for row_number, record in enumerate(records, start=2):
        tokens = _read_json(record["tokens"], "tokens", row_number)
        slot_tags = _read_json(record["slot_tags"], "slot_tags", row_number)
        slot_ids = _read_json(record["slot_ids"], "slot_ids", row_number)
        metadata = _read_json(record["product_metadata"], "product_metadata", row_number)
        generation_info = _read_json(record["generation_info"], "generation_info", row_number)

        if not isinstance(tokens, list) or not tokens or " ".join(tokens) != record["text_query"]:
            raise ValueError(f"Query tokens do not reconstruct text_query at CSV row {row_number}.")
        if len(tokens) != len(slot_tags) or len(tokens) != len(slot_ids):
            raise ValueError(f"Token and slot annotation lengths differ at CSV row {row_number}.")
        if record["intent"] not in INTENT_NAME_TO_ID:
            raise ValueError(f"Unknown intent {record['intent']!r} at CSV row {row_number}.")
        intent_id = int(record["intent_id"])
        if intent_id != INTENT_NAME_TO_ID[record["intent"]]:
            raise ValueError(f"Intent ID does not match intent name at CSV row {row_number}.")
        if any(tag not in SLOT_TAG_TO_ID for tag in slot_tags):
            raise ValueError(f"Unknown BIO slot tag at CSV row {row_number}.")
        if slot_ids != [SLOT_TAG_TO_ID[tag] for tag in slot_tags]:
            raise ValueError(f"Slot IDs do not match BIO tags at CSV row {row_number}.")
        _validate_bio(slot_tags, row_number)
        intent_counts[record["intent"]] += 1
        slot_counts.update(slot_tags)

        template_id = generation_info.get("template_id")
        if template_id not in {
            f"template_{index:02d}" for index in range(len(QUERY_TEMPLATES))
        }:
            raise ValueError(f"Unknown query template at CSV row {row_number}.")
        template_ids.add(template_id)
        if generation_info.get("renderer") != RENDERER_NAME:
            raise ValueError(f"Unexpected image renderer at CSV row {row_number}.")

        product_id = record["product_id"]
        if metadata.get("product_id") != product_id:
            raise ValueError(f"Product metadata ID mismatch at CSV row {row_number}.")
        metadata_key = json.dumps(metadata, sort_keys=True)
        if product_id in products and products[product_id] != metadata_key:
            raise ValueError(f"Product metadata changes between rows for {product_id}.")
        products[product_id] = metadata_key
        product_examples[product_id] += 1
        colors.add(metadata.get("color"))
        shapes.add(metadata.get("shape"))
        if metadata.get("image_path") != record["image_path"]:
            raise ValueError(f"Image path and product metadata disagree at CSV row {row_number}.")
        entity_values = _extract_entities(tokens, slot_tags)
        expected_categories = {
            metadata.get("category_phrase"),
            *metadata.get("compatible_with", ()),
        }
        category_entities = entity_values.get("CATEGORY", set())
        if (
            metadata.get("category_phrase") not in category_entities
            or not category_entities.issubset(expected_categories)
        ):
            raise ValueError(f"Category annotation and product metadata disagree at CSV row {row_number}.")
        required_entities = {
            "COLOR": {metadata.get("color")},
            "SHAPE": {metadata.get("shape")},
        }
        for entity_type, expected_values in required_entities.items():
            if not expected_values.issubset(entity_values.get(entity_type, set())):
                raise ValueError(
                    f"{entity_type} annotation does not match product metadata "
                    f"at CSV row {row_number}."
                )
        if "MATERIAL" in entity_values and entity_values["MATERIAL"] != {
            metadata.get("material")
        }:
            raise ValueError(f"Material annotation and product metadata disagree at CSV row {row_number}.")
        if "PRICE" in entity_values and entity_values["PRICE"] != {
            f"${metadata['price_cents'] / 100:.2f}"
        }:
            raise ValueError(f"Price annotation and product metadata disagree at CSV row {row_number}.")

        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            image_path = csv_path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image for {product_id}: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (224, 224) or image.mode != "RGB":
                raise ValueError(f"Image must be 224x224 RGB: {image_path}")
            image_hashes.add(hashlib.sha256(image.tobytes()).hexdigest())

        key = (product_id, record["text_query"])
        if key in record_keys:
            raise ValueError(f"Duplicate product/query record at CSV row {row_number}.")
        record_keys.add(key)

    if set(intent_counts) != set(INTENT_NAME_TO_ID):
        raise ValueError("Dataset must contain every defined intent.")
    if any(count < 2 for count in intent_counts.values()):
        raise ValueError("Every intent must have multiple examples.")
    if len(template_ids) < 10:
        raise ValueError("Dataset must use at least ten distinct query templates.")
    if len(products) < min_products or min(product_examples.values()) < 2:
        raise ValueError("Insufficient product diversity or examples per product.")
    if len(colors) < 5 or len(shapes) < 5 or len(image_hashes) < 10:
        raise ValueError("Product appearance diversity is too low.")

    if not summary_path.is_file():
        raise FileNotFoundError(f"Dataset summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_summary = {
        "num_examples": len(records),
        "num_unique_products": len(products),
        "num_query_templates": len(template_ids),
        "intent_distribution": dict(sorted(intent_counts.items())),
        "slot_label_distribution": dict(sorted(slot_counts.items())),
        "image_generation_method": RENDERER_NAME,
    }
    for key, expected_value in expected_summary.items():
        if summary.get(key) != expected_value:
            raise ValueError(f"Dataset summary field {key!r} is incorrect.")
    return summary


def _validate_bio(tags, row_number):
    previous_type = None
    for tag in tags:
        if tag == "O":
            previous_type = None
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "B":
            previous_type = entity_type
        elif prefix == "I" and previous_type == entity_type:
            previous_type = entity_type
        else:
            raise ValueError(f"Invalid BIO transition at CSV row {row_number}: {tag}.")


def _extract_entities(tokens, tags):
    entities = {}
    active_type = None
    active_tokens = []
    for token, tag in zip(tokens + ["O"], tags + ["O"]):
        if tag == "O":
            if active_type is not None:
                entities.setdefault(active_type, set()).add(" ".join(active_tokens))
                active_type = None
                active_tokens = []
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "B" or entity_type != active_type:
            if active_type is not None:
                entities.setdefault(active_type, set()).add(" ".join(active_tokens))
            active_type = entity_type
            active_tokens = [token]
        else:
            active_tokens.append(token)
    return entities


def validate_reproducibility(seed: int = 42) -> None:
    """Regenerate a compact catalogue twice and compare all output bytes."""
    with tempfile.TemporaryDirectory(prefix="synthetic-search-repro-") as temporary_dir:
        root = Path(temporary_dir)
        first_dir = root / "first"
        second_dir = root / "second"
        generate_synthetic_dataset(first_dir, seed=seed, num_products=10)
        generate_synthetic_dataset(second_dir, seed=seed, num_products=10)

        for relative_path in (
            Path("dataset.csv"),
            Path("catalogue.json"),
            Path("dataset_summary.json"),
        ):
            if (first_dir / relative_path).read_bytes() != (second_dir / relative_path).read_bytes():
                raise ValueError(f"Fixed-seed generation differs for {relative_path}.")
        for image_path in sorted((first_dir / "images").glob("*.png")):
            other_image = second_dir / "images" / image_path.name
            if not other_image.is_file() or image_path.read_bytes() != other_image.read_bytes():
                raise ValueError(f"Fixed-seed image generation differs for {image_path.name}.")


def validate_wordpiece_masking() -> None:
    """Exercise loader alignment using a tiny local WordPiece vocabulary."""
    import csv
    import tempfile

    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertTokenizerFast

    from src.dataset import MultiModalDataset

    vocabulary = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "find", "un", "##break", "##able")

    class StubImageProcessor:
        def __call__(self, images, return_tensors):
            return {"pixel_values": torch.zeros((1, 3, 224, 224))}

    with tempfile.TemporaryDirectory(prefix="synthetic-search-wordpiece-") as temporary_dir:
        root = Path(temporary_dir)
        images_dir = root / "images"
        images_dir.mkdir()
        Image.new("RGB", (224, 224), (220, 30, 30)).save(images_dir / "probe.png")
        vocabulary_ids = {token: token_id for token_id, token in enumerate(vocabulary)}
        wordpiece_tokenizer = Tokenizer(
            models.WordPiece(vocab=vocabulary_ids, unk_token="[UNK]")
        )
        wordpiece_tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        wordpiece_tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        wordpiece_tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[
                ("[CLS]", vocabulary_ids["[CLS]"]),
                ("[SEP]", vocabulary_ids["[SEP]"]),
            ],
        )
        tokenizer = BertTokenizerFast(tokenizer_object=wordpiece_tokenizer)
        dataset_path = root / "dataset.csv"
        record = {
            "image_path": "images/probe.png",
            "text_query": "find unbreakable",
            "intent_id": 0,
            "slot_ids": json.dumps([0, SLOT_TAG_TO_ID["B-CATEGORY"]]),
            "tokens": json.dumps(["find", "unbreakable"]),
            "slot_tags": json.dumps(["O", "B-CATEGORY"]),
            "intent": "search_product",
        }
        with dataset_path.open("w", newline="", encoding="utf-8") as dataset_file:
            writer = csv.DictWriter(dataset_file, fieldnames=record)
            writer.writeheader()
            writer.writerow(record)

        dataset = MultiModalDataset(
            dataset_path,
            max_length=8,
            tokenizer=tokenizer,
            image_processor=StubImageProcessor(),
        )
        sample = dataset[0]
        actual_labels = sample["slot_labels"].tolist()
        expected_labels = [-100, 0, 1, 2, 2, -100, -100, -100]
        if actual_labels != expected_labels:
            raise ValueError(f"WordPiece/special/padding alignment failed: {actual_labels}.")
        if sample["attention_mask"].tolist() != [1, 1, 1, 1, 1, 1, 0, 0]:
            raise ValueError("Tokenizer special-token or padding attention mask is incorrect.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "raw" / "dataset.csv",
    )
    parser.add_argument("--min-examples", type=int, default=600)
    args = parser.parse_args()
    summary = validate_dataset(args.csv, min_examples=args.min_examples)
    validate_reproducibility(seed=int(summary["seed"]))
    validate_wordpiece_masking()
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Validation passed: images, labels, alignment, diversity, uniqueness, and reproducibility.")


if __name__ == "__main__":
    main()
