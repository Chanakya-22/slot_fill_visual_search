from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer


class MultiModalDataset(Dataset):
    def __init__(self, dataframe_path: str | Path) -> None:
        self.dataframe_path = Path(dataframe_path).resolve()
        self.dataframe = pd.read_csv(self.dataframe_path)
        self.project_root = self.dataframe_path.parents[2]

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _parse_slot_ids(self, slot_ids: Any) -> List[int]:
        if isinstance(slot_ids, str):
            slot_ids = ast.literal_eval(slot_ids)
        if isinstance(slot_ids, (list, tuple)):
            return [int(value) for value in slot_ids]
        return [0] * 32

    def _align_slot_labels(self, text: str, slot_ids: List[int], encoding: Dict[str, Any]) -> torch.Tensor:
        tokenized_words = text.split()
        word_slots = []
        for word_idx, _ in enumerate(tokenized_words):
            if word_idx < len(slot_ids):
                word_slots.append(int(slot_ids[word_idx]))
            else:
                word_slots.append(0)

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels: List[int] = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(0)
            else:
                aligned_labels.append(word_slots[word_id] if word_id < len(word_slots) else 0)

        if len(aligned_labels) < 64:
            aligned_labels += [0] * (64 - len(aligned_labels))
        else:
            aligned_labels = aligned_labels[:64]

        return torch.tensor(aligned_labels, dtype=torch.long)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]

        text_query = str(row["text_query"])
        image_path_value = str(row["image_path"])
        image_path = self.project_root / image_path_value
        intent_label = int(row["intent_id"])
        slot_ids = self._parse_slot_ids(row["slot_ids"])

        image = Image.open(image_path).convert("RGB")

        text_encoding = self.tokenizer(
            text_query,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )

        image_encoding = self.processor(images=image, return_tensors="pt")

        slot_labels = self._align_slot_labels(text_query, slot_ids, text_encoding)

        return {
            "input_ids": text_encoding["input_ids"].squeeze(0).long(),
            "attention_mask": text_encoding["attention_mask"].squeeze(0).long(),
            "pixel_values": image_encoding["pixel_values"].squeeze(0).float(),
            "intent_label": torch.tensor(intent_label, dtype=torch.long),
            "slot_labels": slot_labels,
        }
