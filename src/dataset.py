import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoProcessor

class MultiModalDataset(Dataset):
    def __init__(
        self,
        csv_path,
        text_model_name="bert-base-uncased",
        vision_model_name="openai/clip-vit-base-patch32",
        max_length=32
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.image_processor = AutoProcessor.from_pretrained(vision_model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load & Process Text
        text = str(row["text_query"])
        text_encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 2. Load & Process Image
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_inputs = self.image_processor(images=image, return_tensors="pt")

        # 3. Dynamic Sub-Word Token Alignment
        raw_slots = json.loads(row["slot_ids"]) if isinstance(row["slot_ids"], str) else row["slot_ids"]
        word_ids = text_encoding.word_ids(batch_index=0)
        
        aligned_slots = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens like [CLS], [SEP], [PAD] get -100
                aligned_slots.append(-100)
            elif word_idx < len(raw_slots):
                # Map the sub-word to its parent word's slot label
                aligned_slots.append(raw_slots[word_idx])
            else:
                aligned_slots.append(-100)

        return {
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "intent_label": torch.tensor(int(row["intent_id"]), dtype=torch.long),
            "slot_labels": torch.tensor(aligned_slots, dtype=torch.long)
        }