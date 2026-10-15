import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer

from src.labels import INTENT_ID_TO_NAME, INTENT_NAME_TO_ID, SLOT_TAG_TO_ID

LEGACY_SLOT_ID_TO_TAG = {0: "O", 1: "B-CATEGORY", 2: "B-MATERIAL"}


class MultiModalDataset(Dataset):
    def __init__(
        self,
        csv_path,
        text_model_name="bert-base-uncased",
        vision_model_name="openai/clip-vit-base-patch32",
        max_length=32,
        tokenizer=None,
        image_processor=None,
    ):
        self.csv_path = Path(csv_path).resolve()
        self.df = pd.read_csv(self.csv_path)
        required_columns = {"image_path", "text_query", "intent_id", "slot_ids"}
        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(text_model_name)
        self.image_processor = image_processor or AutoProcessor.from_pretrained(vision_model_name)
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is required for word-to-WordPiece alignment.")
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def _read_annotations(self, row):
        text = str(row["text_query"])
        tokens_value = row.get("tokens")
        tokens = json.loads(tokens_value) if isinstance(tokens_value, str) and tokens_value else text.split()
        if not tokens or " ".join(tokens) != text:
            raise ValueError(f"Stored tokens do not match text_query for row {row.name}.")

        tags_value = row.get("slot_tags")
        if isinstance(tags_value, str) and tags_value:
            slot_tags = json.loads(tags_value)
            unknown_tags = sorted(set(slot_tags).difference(SLOT_TAG_TO_ID))
            if unknown_tags:
                raise ValueError(f"Unknown BIO slot tags in row {row.name}: {unknown_tags}")
            raw_slot_ids = json.loads(row["slot_ids"]) if isinstance(row["slot_ids"], str) else row["slot_ids"]
            if [int(slot_id) for slot_id in raw_slot_ids] != [
                SLOT_TAG_TO_ID[tag] for tag in slot_tags
            ]:
                raise ValueError(f"BIO slot tags and IDs disagree in row {row.name}.")
        else:
            legacy_ids = json.loads(row["slot_ids"]) if isinstance(row["slot_ids"], str) else row["slot_ids"]
            try:
                slot_tags = [LEGACY_SLOT_ID_TO_TAG[int(slot_id)] for slot_id in legacy_ids]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid legacy slot IDs in row {row.name}.") from exc

        if len(tokens) != len(slot_tags):
            raise ValueError(
                f"Token/slot length mismatch in row {row.name}: {len(tokens)} tokens, "
                f"{len(slot_tags)} labels."
            )
        previous_entity_type = None
        for tag in slot_tags:
            if tag == "O":
                previous_entity_type = None
                continue
            prefix, entity_type = tag.split("-", 1)
            if prefix == "B":
                previous_entity_type = entity_type
            elif prefix == "I" and previous_entity_type == entity_type:
                previous_entity_type = entity_type
            else:
                raise ValueError(
                    f"Invalid BIO transition in row {row.name}: {tag}."
                )
        return text, tokens, slot_tags

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text, tokens, slot_tags = self._read_annotations(row)
        text_encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = text_encoding.word_ids(batch_index=0)
        aligned_slots = []
        previous_word_id = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_slots.append(-100)
            else:
                tag = slot_tags[word_idx]
                if word_idx == previous_word_id and tag.startswith("B-"):
                    tag = f"I-{tag[2:]}"
                aligned_slots.append(SLOT_TAG_TO_ID[tag])
            previous_word_id = word_idx

        image_path = Path(str(row["image_path"]))
        if "product_id" in row.index:
            product_id = row["product_id"]
            if (
                not isinstance(product_id, str)
                or re.fullmatch(r"product_\d{4}", product_id) is None
            ):
                raise ValueError(
                    f"Non-canonical product ID in row {row.name}: {product_id!r}."
                )
            canonical_path = self.csv_path.parent / "images" / f"{product_id}.png"
            if image_path != Path("images") / f"{product_id}.png":
                raise ValueError(
                    f"Product image path in row {row.name} must be canonical: "
                    f"images/{product_id}.png."
                )
            image_path = canonical_path
        if not image_path.is_absolute():
            image_path = self.csv_path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image for row {row.name} does not exist: {image_path}")
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        image_inputs = self.image_processor(images=image, return_tensors="pt")

        intent_id = int(row["intent_id"])
        if intent_id not in INTENT_ID_TO_NAME:
            raise ValueError(f"Unknown intent ID {intent_id} in row {row.name}.")
        if "intent" in row.index and isinstance(row["intent"], str):
            expected_id = INTENT_NAME_TO_ID.get(row["intent"])
            if expected_id is None:
                raise ValueError(f"Unknown intent {row['intent']!r} in row {row.name}.")
            if intent_id != expected_id:
                raise ValueError(f"Intent name and ID disagree in row {row.name}.")

        return {
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "intent_label": torch.tensor(intent_id, dtype=torch.long),
            "slot_labels": torch.tensor(aligned_slots, dtype=torch.long),
        }
