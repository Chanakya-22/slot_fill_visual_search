import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from src.labels import SLOT_TAG_TO_ID
from src.synthetic_data import CATEGORIES, COLORS, generate_synthetic_dataset, render_product_image
from src.utils import compute_slot_metrics
from train import split_dataset
from validate_synthetic_dataset import (
    _validate_bio,
    count_unreferenced_normalized_image_aliases,
    validate_dataset,
    validate_reproducibility,
    validate_wordpiece_masking,
)


class SyntheticDatasetTests(unittest.TestCase):
    def test_compact_dataset_schema_images_and_summary(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-search-test-") as temporary_dir:
            output_dir = Path(temporary_dir)
            summary = generate_synthetic_dataset(output_dir, seed=17, num_products=10)
            self.assertEqual(summary["num_examples"], 100)
            self.assertEqual(summary["num_unique_products"], 10)
            self.assertEqual(summary["num_query_templates"], 24)
            self.assertGreater(summary["slot_label_distribution"]["B-MATERIAL"], 0)
            self.assertGreater(summary["slot_label_distribution"]["B-PRICE"], 0)
            legacy_alias = output_dir / "images" / "product_000.png"
            Image.new("RGB", (224, 224), (2, 3, 4)).save(legacy_alias)
            validated = validate_dataset(
                output_dir / "dataset.csv", min_examples=50, min_products=5
            )
            self.assertEqual(validated["seed"], 17)
            self.assertEqual(
                count_unreferenced_normalized_image_aliases(output_dir / "dataset.csv"),
                1,
            )
            with Image.open(legacy_alias) as image:
                self.assertEqual(image.getpixel((0, 0)), (2, 3, 4))

            with (output_dir / "dataset.csv").open(newline="", encoding="utf-8") as dataset_file:
                records = list(csv.DictReader(dataset_file))
            self.assertEqual(len({record["sample_id"] for record in records}), len(records))
            self.assertEqual(
                len(
                    {
                        (
                            record["product_id"],
                            record["text_query"],
                            record["intent"],
                        )
                        for record in records
                    }
                ),
                len(records),
            )
            for record in records:
                self.assertRegex(record["product_id"], r"^product_\d{4}$")
                self.assertEqual(
                    record["image_path"],
                    f"images/{record['product_id']}.png",
                )
                tags = json.loads(record["slot_tags"])
                ids = json.loads(record["slot_ids"])
                self.assertEqual(len(tags), len(json.loads(record["tokens"])))
                self.assertEqual(ids, [SLOT_TAG_TO_ID[tag] for tag in tags])
                image_path = output_dir / record["image_path"]
                with Image.open(image_path) as image:
                    self.assertEqual(image.size, (224, 224))
                    self.assertEqual(image.mode, "RGB")

    def test_category_shape_variants_render_distinct_images(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-shapes-") as temporary_dir:
            output_dir = Path(temporary_dir)
            for category, specification in CATEGORIES.items():
                image_hashes = set()
                for shape_index, shape in enumerate(specification["shapes"]):
                    image_path = output_dir / f"{category}_{shape_index}.png"
                    render_product_image(
                        {
                            "category": category,
                            "shape_index": shape_index,
                            "shape": shape,
                            "color": "red",
                        },
                        image_path,
                    )
                    with Image.open(image_path) as image:
                        self.assertGreater(
                            Counter(image.convert("RGB").get_flattened_data())[COLORS["red"]],
                            100,
                        )
                        image_hashes.add(
                            hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
                        )
                self.assertEqual(
                    len(image_hashes),
                    len(specification["shapes"]),
                    f"{category} shape variants should render distinctly",
                )

    def test_fixed_seed_reproduces_csv_catalogue_and_images(self):
        validate_reproducibility(seed=23)

    def test_wordpiece_alignment_masks_special_tokens_and_padding(self):
        validate_wordpiece_masking()

    def test_slot_f1_is_entity_level_and_ignores_masked_positions(self):
        metrics = compute_slot_metrics(
            preds=[[0, 1, 0, 9]],
            labels=[[0, 1, 2, -100]],
        )
        self.assertEqual(metrics["true_entity_count"], 1)
        self.assertEqual(metrics["predicted_entity_count"], 1)
        self.assertEqual(metrics["slot_f1"], 0.0)

    def test_invalid_bio_entity_boundary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid BIO transition"):
            _validate_bio(["O", "I-COLOR"], row_number=1)

    def test_seeded_product_group_split_has_no_product_leakage(self):
        class MetadataDataset:
            def __init__(self, frame):
                self.df = frame

            def __len__(self):
                return len(self.df)

        records = []
        for product_index in range(10):
            for repeat in range(2):
                for intent_id in range(5):
                    records.append(
                        {
                            "sample_id": f"sample_{product_index}_{repeat}_{intent_id}",
                            "product_id": f"product_{product_index:04d}",
                            "intent_id": intent_id,
                            "generation_info": json.dumps(
                                {
                                    "template_id": (
                                        f"template_{intent_id * 4 + (product_index + repeat) % 4:02d}"
                                    )
                                }
                            ),
                        }
                    )
        frame = pd.DataFrame(records)
        dataset = MetadataDataset(frame)
        train_set, validation_set, test_set, details = split_dataset(dataset, seed=42)
        train_products = set(frame.iloc[train_set.indices]["product_id"])
        validation_products = set(frame.iloc[validation_set.indices]["product_id"])
        test_products = set(frame.iloc[test_set.indices]["product_id"])
        self.assertFalse(train_products & validation_products)
        self.assertFalse(train_products & test_products)
        self.assertFalse(validation_products & test_products)
        self.assertEqual(sum(map(len, (train_set, validation_set, test_set))), len(frame))
        for subset in (train_set, validation_set, test_set):
            self.assertEqual(set(frame.iloc[subset.indices]["intent_id"]), set(range(5)))

        repeated_train, repeated_validation, repeated_test, repeated_details = split_dataset(
            dataset, seed=42
        )
        self.assertEqual(train_set.indices, repeated_train.indices)
        self.assertEqual(validation_set.indices, repeated_validation.indices)
        self.assertEqual(test_set.indices, repeated_test.indices)
        self.assertEqual(details, repeated_details)

        template_splits = split_dataset(
            dataset, seed=42, split_mode="template"
        )
        template_groups = [
            set(
                json.loads(value)["template_id"]
                for value in frame.iloc[subset.indices]["generation_info"]
            )
            for subset in template_splits[:3]
        ]
        self.assertFalse(template_groups[0] & template_groups[1])
        self.assertFalse(template_groups[0] & template_groups[2])
        self.assertFalse(template_groups[1] & template_groups[2])


if __name__ == "__main__":
    unittest.main()
