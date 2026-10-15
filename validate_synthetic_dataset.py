"""Validate the generated synthetic visual-search catalogue without model downloads."""

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image

from src.labels import INTENT_NAME_TO_ID, SLOT_TAG_TO_ID
from src.synthetic_data import (
    CATEGORIES,
    COLORS,
    QUERY_TEMPLATES,
    RENDERER_NAME,
    _render_query,
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
    catalogue_path = csv_path.parent / "catalogue.json"
    if not catalogue_path.is_file():
        raise FileNotFoundError(f"Product catalogue not found: {catalogue_path}")
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    if not isinstance(catalogue, list):
        raise ValueError("Product catalogue must contain a JSON list.")
    catalogue_by_id = {}
    normalized_catalogue_ids = {}
    for product in catalogue:
        product_id = product.get("product_id")
        if not isinstance(product_id, str) or not re.fullmatch(r"product_\d{4}", product_id):
            raise ValueError(f"Product ID is not in canonical product_NNNN form: {product_id!r}")
        normalized_id = int(product_id.split("_", 1)[1])
        if normalized_id in normalized_catalogue_ids:
            raise ValueError(
                f"Normalized product ID collision: {normalized_catalogue_ids[normalized_id]} "
                f"and {product_id}."
            )
        normalized_catalogue_ids[normalized_id] = product_id
        if product_id in catalogue_by_id:
            raise ValueError(f"Duplicate catalogue product ID: {product_id}.")
        catalogue_by_id[product_id] = product

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
    image_products_by_hash = {}
    colors = set()
    shapes = set()
    entity_row_counts = Counter()

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
        template_match = re.fullmatch(r"template_(\d{2})", str(template_id))
        if template_match is None or int(template_match.group(1)) >= len(QUERY_TEMPLATES):
            raise ValueError(f"Unknown query template at CSV row {row_number}.")
        template_index = int(template_match.group(1))
        template_intent, template = QUERY_TEMPLATES[template_index]
        if template_intent != record["intent"]:
            raise ValueError(f"Intent and query template disagree at CSV row {row_number}.")
        template_ids.add(template_id)
        if generation_info.get("renderer") != RENDERER_NAME:
            raise ValueError(f"Unexpected image renderer at CSV row {row_number}.")

        product_id = record["product_id"]
        if not re.fullmatch(r"product_\d{4}", product_id):
            raise ValueError(f"CSV product ID is not canonical at row {row_number}: {product_id!r}")
        if metadata.get("product_id") != product_id:
            raise ValueError(f"Product metadata ID mismatch at CSV row {row_number}.")
        catalogue_product = catalogue_by_id.get(product_id)
        if catalogue_product is None:
            raise ValueError(f"CSV product {product_id} is absent from catalogue.json.")
        if metadata != catalogue_product:
            raise ValueError(f"CSV metadata does not match catalogue.json for {product_id}.")
        if record["image_path"] != f"images/{product_id}.png":
            raise ValueError(f"CSV image path is not canonical for {product_id}.")
        category = metadata.get("category")
        shape_index = metadata.get("shape_index")
        if category not in CATEGORIES or not isinstance(shape_index, int):
            raise ValueError(f"Invalid category or shape index for {product_id}.")
        if not 0 <= shape_index < len(CATEGORIES[category]["shapes"]):
            raise ValueError(f"Shape index is out of range for {product_id}.")
        if metadata.get("shape") != CATEGORIES[category]["shapes"][shape_index]:
            raise ValueError(f"Shape metadata does not match category for {product_id}.")

        template_fields = {
            "category": metadata["category_phrase"],
            "compatible_category": CATEGORIES[metadata["compatible_with"][0]]["phrases"][0],
            "color": metadata["color"],
            "shape": metadata["shape"],
            "material": metadata["material"],
            "price": f"${metadata['price_cents'] / 100:.2f}",
        }
        expected_tokens, expected_tags = _render_query(template, template_fields)
        if tokens != expected_tokens or slot_tags != expected_tags:
            raise ValueError(
                f"Query tokens or BIO boundaries do not match template {template_id} "
                f"at CSV row {row_number}."
            )

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
        for entity_type in entity_values:
            entity_row_counts[(record["intent"], entity_type)] += 1
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
            image_rgb = image.convert("RGB")
            image_hash = hashlib.sha256(image_rgb.tobytes()).hexdigest()
            image_hashes.add(image_hash)
            image_products_by_hash.setdefault(image_hash, set()).add(product_id)
            color_counts = Counter(image_rgb.get_flattened_data())
        expected_color = COLORS.get(metadata.get("color"))
        if expected_color is None or color_counts[expected_color] < 100:
            raise ValueError(
                f"Rendered image does not contain its recorded finish color for {product_id}."
            )

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
    if set(products) != set(catalogue_by_id):
        raise ValueError("CSV product IDs and catalogue product IDs do not match exactly.")
    if len(image_hashes) != len(products):
        collisions = [
            sorted(product_ids)
            for product_ids in image_products_by_hash.values()
            if len(product_ids) > 1
        ]
        raise ValueError(f"Distinct catalogue products share rendered images: {collisions[:5]}.")
    if not entity_row_counts[("request_specifications", "MATERIAL")]:
        raise ValueError("No request_specifications examples contain material entities.")
    if not entity_row_counts[("request_specifications", "PRICE")]:
        raise ValueError("No request_specifications examples contain price entities.")
    if any(
        count
        for (intent, entity_type), count in entity_row_counts.items()
        if entity_type in {"MATERIAL", "PRICE"} and intent != "request_specifications"
    ):
        raise ValueError("Material and price entities must appear only in specification requests.")

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


def count_unreferenced_normalized_image_aliases(csv_path: Path | str) -> int:
    """Count old product image names that collide numerically but are not referenced."""
    import csv

    csv_path = Path(csv_path).resolve()
    with csv_path.open(newline="", encoding="utf-8") as dataset_file:
        referenced_names = {
            Path(record["image_path"]).name
            for record in csv.DictReader(dataset_file)
        }
    canonical_ids = {
        int(match.group(1))
        for name in referenced_names
        if (match := re.fullmatch(r"product_(\d{4})\.png", name))
    }
    image_dir = csv_path.parent / "images"
    aliases = set()
    if image_dir.is_dir():
        for image_path in image_dir.glob("product_*.png"):
            match = re.fullmatch(r"product_(\d+)\.png", image_path.name)
            if (
                match is not None
                and image_path.name not in referenced_names
                and int(match.group(1)) in canonical_ids
            ):
                aliases.add(image_path.name)
    return len(aliases)


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
    legacy_aliases = count_unreferenced_normalized_image_aliases(args.csv)
    if legacy_aliases:
        print(
            f"Note: found {legacy_aliases} unreferenced legacy image filenames "
            "with numeric IDs matching canonical products; left them untouched."
        )
    print("Validation passed: images, labels, alignment, diversity, uniqueness, and reproducibility.")


if __name__ == "__main__":
    main()
