import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.labels import NUM_INTENTS, NUM_SLOT_LABELS
from src.models.baseline_concat import BaselineConcatModel
from src.models.baseline_text import BaselineTextModel
from src.models.cross_attention_model import CrossAttentionModel
from train import evaluate, train_single_model


class TinyBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(32, 8)

    @classmethod
    def from_pretrained(cls, _model_name):
        return cls()

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return SimpleNamespace(pooler_output=pooled, last_hidden_state=hidden)


class TinyClipEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])


class TinyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.projection = nn.Linear(3, 8)
        self.encoder = TinyClipEncoder()

    @classmethod
    def from_pretrained(cls, _model_name):
        return cls()

    def forward(self, pixel_values):
        image_mean = pixel_values.mean(dim=(2, 3))
        hidden = self.projection(image_mean).unsqueeze(1)
        for layer in self.encoder.layers:
            hidden = torch.tanh(layer(hidden))
        patches = hidden.expand(-1, 5, -1)
        return SimpleNamespace(pooler_output=hidden.squeeze(1), last_hidden_state=patches)


class ModelSmokeTests(unittest.TestCase):
    def test_each_architecture_trains_evaluates_saves_reloads_and_predicts(self):
        dataset = [
            {
                "input_ids": torch.tensor([1, 2, 3, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 0]),
                "pixel_values": torch.full((3, 8, 8), index / 4),
                "intent_label": torch.tensor(index % NUM_INTENTS),
                "slot_labels": torch.tensor([0, 1, 2, -100]),
            }
            for index in range(4)
        ]
        loader = DataLoader(dataset, batch_size=2)
        architectures = (
            ("BaselineTextModel", BaselineTextModel, "src.models.baseline_text"),
            ("BaselineConcatModel", BaselineConcatModel, "src.models.baseline_concat"),
            ("CrossAttentionModel", CrossAttentionModel, "src.models.cross_attention_model"),
        )

        for model_name, model_type, module_name in architectures:
            with self.subTest(model=model_name):
                with ExitStack() as patches:
                    patches.enter_context(patch(f"{module_name}.BertModel", TinyBert))
                    if model_type is not BaselineTextModel:
                        patches.enter_context(
                            patch(f"{module_name}.CLIPVisionModel", TinyClip)
                        )
                    model = model_type(
                        num_intents=NUM_INTENTS,
                        num_slots=NUM_SLOT_LABELS,
                    )
                    batch = next(iter(loader))
                    if model_type is BaselineTextModel:
                        logits = model(batch["input_ids"], batch["attention_mask"])
                    else:
                        logits = model(
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["pixel_values"],
                        )
                    self.assertTrue(all(torch.isfinite(output).all() for output in logits))
                    (logits[0].sum() + logits[1].sum()).backward()
                    self.assertTrue(
                        any(
                            parameter.grad is not None
                            and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        )
                    )

                    with tempfile.TemporaryDirectory(
                        prefix="model-smoke-checkpoint-"
                    ) as temporary_dir:
                        checkpoint_path = Path(temporary_dir) / "smoke.pt"
                        trained = train_single_model(
                            model_name=model_name,
                            model=model_type(
                                num_intents=NUM_INTENTS,
                                num_slots=NUM_SLOT_LABELS,
                            ),
                            train_loader=loader,
                            validation_loader=loader,
                            device=torch.device("cpu"),
                            seed=42,
                            split_mode="product",
                            epochs=1,
                            patience=1,
                            checkpoint_path=checkpoint_path,
                        )
                        self.assertTrue(torch.isfinite(torch.tensor(
                            trained["validation_metrics"]["intent"]["macro_f1"]
                        )))
                        self.assertTrue(torch.isfinite(torch.tensor(
                            trained["validation_metrics"]["slots"]["slot_f1"]
                        )))
                        saved = torch.load(checkpoint_path, map_location="cpu")
                        reloaded = model_type(
                            num_intents=NUM_INTENTS,
                            num_slots=NUM_SLOT_LABELS,
                        )
                        reloaded.load_state_dict(saved["model_state_dict"])
                        metrics = evaluate(reloaded, loader, torch.device("cpu"))
                        self.assertTrue(
                            torch.isfinite(
                                torch.tensor(metrics["intent"]["accuracy"])
                            )
                        )
                        self.assertTrue(
                            torch.isfinite(torch.tensor(metrics["intent"]["macro_f1"]))
                        )
                        self.assertTrue(
                            torch.isfinite(torch.tensor(metrics["slots"]["slot_f1"]))
                        )
                        predictions, slot_predictions = [], []
                        reloaded.eval()
                        with torch.no_grad():
                            for inference_batch in loader:
                                if model_type is BaselineTextModel:
                                    outputs = reloaded(
                                        inference_batch["input_ids"],
                                        inference_batch["attention_mask"],
                                    )
                                else:
                                    outputs = reloaded(
                                        inference_batch["input_ids"],
                                        inference_batch["attention_mask"],
                                        inference_batch["pixel_values"],
                                    )
                                predictions.extend(outputs[0].argmax(-1).tolist())
                                slot_predictions.extend(
                                    outputs[1].argmax(-1).tolist()
                                )
                        self.assertTrue(
                            all(0 <= prediction < NUM_INTENTS for prediction in predictions)
                        )
                        self.assertTrue(
                            all(
                                0 <= prediction < NUM_SLOT_LABELS
                                for row in slot_predictions
                                for prediction in row
                            )
                        )

    def test_cross_attention_uses_image_features(self):
        with patch("src.models.cross_attention_model.BertModel", TinyBert), patch(
            "src.models.cross_attention_model.CLIPVisionModel", TinyClip
        ):
            model = CrossAttentionModel(
                num_intents=NUM_INTENTS,
                num_slots=NUM_SLOT_LABELS,
            ).eval()
            input_ids = torch.tensor([[1, 2, 3, 0]])
            attention_mask = torch.tensor([[1, 1, 1, 0]])
            image_a = torch.zeros((1, 3, 8, 8))
            image_b = torch.ones((1, 3, 8, 8))
            with torch.no_grad():
                output_a = model(input_ids, attention_mask, image_a)
                output_b = model(input_ids, attention_mask, image_b)
            self.assertFalse(torch.allclose(output_a[0], output_b[0]))
            self.assertFalse(torch.allclose(output_a[1], output_b[1]))


if __name__ == "__main__":
    unittest.main()
