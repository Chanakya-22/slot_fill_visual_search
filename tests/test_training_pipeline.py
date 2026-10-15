import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train import train_single_model


class TinyMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.intent_head = nn.Linear(2, 5)
        self.slot_head = nn.Linear(2, 11)

    def forward(self, input_ids, attention_mask, pixel_values):
        _ = attention_mask, pixel_values
        features = input_ids.float()
        intent_logits = self.intent_head(features)
        slot_logits = self.slot_head(features).unsqueeze(1).expand(
            -1, input_ids.shape[1], -1
        )
        return intent_logits, slot_logits


class TrainingPipelineTests(unittest.TestCase):
    def test_best_validation_checkpoint_is_persisted_with_metadata(self):
        dataset = [
            {
                "input_ids": torch.tensor([float(index), 1.0]),
                "attention_mask": torch.ones(2),
                "pixel_values": torch.zeros(3),
                "intent_label": torch.tensor(index),
                "slot_labels": torch.tensor([0, 0]),
            }
            for index in range(5)
        ]
        loader = DataLoader(dataset, batch_size=5)
        with tempfile.TemporaryDirectory(prefix="training-checkpoint-test-") as temp_dir:
            checkpoint = Path(temp_dir) / "best.pt"
            result = train_single_model(
                model_name="TinyMultimodalModel",
                model=TinyMultimodalModel(),
                train_loader=loader,
                validation_loader=loader,
                device=torch.device("cpu"),
                seed=42,
                split_mode="product",
                epochs=3,
                patience=1,
                checkpoint_path=checkpoint,
            )
            self.assertEqual(result["best_epoch"], 1)
            self.assertGreater(result["training_time_seconds"], 0)
            self.assertTrue(checkpoint.is_file())
            saved = torch.load(checkpoint, map_location="cpu")
            self.assertIn("model_state_dict", saved)
            self.assertEqual(saved["metadata"]["seed"], 42)
            self.assertEqual(saved["metadata"]["split_mode"], "product")
            self.assertEqual(saved["metadata"]["best_epoch"], 1)
            self.assertEqual(
                saved["metadata"]["checkpoint_selection_metric"],
                "validation.slots.slot_f1",
            )
            self.assertGreater(saved["metadata"]["training_time_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
