import csv
import hashlib
import json
import re
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

        frame = pd.DataFrame(
            {"product_id": [f"product_{index // 10}" for index in range(100)]}
        )
        dataset = MetadataDataset(frame)
        train_set, validation_set, grouped = split_dataset(dataset, seed=42)
        train_products = set(frame.iloc[train_set.indices]["product_id"])
        validation_products = set(frame.iloc[validation_set.indices]["product_id"])
        self.assertTrue(grouped)
        self.assertFalse(train_products & validation_products)
        self.assertEqual(len(validation_set), 20)

        _, repeated_validation_set, _ = split_dataset(dataset, seed=42)
        self.assertEqual(validation_set.indices, repeated_validation_set.indices)


if __name__ == "__main__":
    unittest.main()
