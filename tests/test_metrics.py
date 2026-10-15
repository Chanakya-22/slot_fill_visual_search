import unittest

from src.labels import INTENT_NAME_TO_ID, SLOT_TAG_TO_ID
from src.utils import compute_intent_metrics, compute_slot_metrics


class MetricTests(unittest.TestCase):
    def test_intent_accuracy_macro_f1_and_per_class_scores(self):
        metrics = compute_intent_metrics(
            preds=[0, 1, 1, 3],
            labels=[0, 1, 2, 4],
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)
        self.assertEqual(len(metrics["class_metrics"]), len(INTENT_NAME_TO_ID))
        self.assertEqual(
            metrics["class_metrics"]["troubleshoot_product"],
            {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 1},
        )
        self.assertEqual(
            metrics["class_metrics"]["identify_product"],
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
        )

    def test_perfect_entity_predictions(self):
        tags = ["O", "B-CATEGORY", "I-CATEGORY", "B-COLOR", "I-COLOR"]
        ids = [SLOT_TAG_TO_ID[tag] for tag in tags]
        metrics = compute_slot_metrics([ids], [ids])
        self.assertEqual(metrics["slot_precision"], 1.0)
        self.assertEqual(metrics["slot_recall"], 1.0)
        self.assertEqual(metrics["slot_f1"], 1.0)
        self.assertEqual(metrics["true_entity_count"], 2)
        self.assertEqual(metrics["predicted_entity_count"], 2)

    def test_missing_entity(self):
        metrics = compute_slot_metrics(
            [[SLOT_TAG_TO_ID["O"]]],
            [[SLOT_TAG_TO_ID["B-CATEGORY"]]],
        )
        self.assertEqual(metrics["slot_precision"], 0.0)
        self.assertEqual(metrics["slot_recall"], 0.0)
        self.assertEqual(metrics["slot_f1"], 0.0)

    def test_incorrect_entity_type_is_not_a_match(self):
        metrics = compute_slot_metrics(
            [[SLOT_TAG_TO_ID["B-COLOR"]]],
            [[SLOT_TAG_TO_ID["B-CATEGORY"]]],
        )
        self.assertEqual(metrics["slot_f1"], 0.0)

    def test_incorrect_entity_boundary_is_not_a_match(self):
        metrics = compute_slot_metrics(
            [[SLOT_TAG_TO_ID["B-CATEGORY"], SLOT_TAG_TO_ID["O"]]],
            [[SLOT_TAG_TO_ID["B-CATEGORY"], SLOT_TAG_TO_ID["I-CATEGORY"]]],
        )
        self.assertEqual(metrics["slot_f1"], 0.0)

    def test_special_tokens_and_padding_are_ignored_and_break_sequences(self):
        metrics = compute_slot_metrics(
            [[SLOT_TAG_TO_ID["B-CATEGORY"], 999, SLOT_TAG_TO_ID["I-CATEGORY"], 999]],
            [[SLOT_TAG_TO_ID["B-CATEGORY"], -100, SLOT_TAG_TO_ID["I-CATEGORY"], -100]],
        )
        self.assertEqual(metrics["slot_f1"], 1.0)
        self.assertEqual(metrics["true_entity_count"], 2)
        self.assertEqual(metrics["predicted_entity_count"], 2)

    def test_empty_entity_sets_have_zero_scores(self):
        metrics = compute_slot_metrics([[SLOT_TAG_TO_ID["O"]]], [[SLOT_TAG_TO_ID["O"]]])
        self.assertEqual(metrics["slot_precision"], 0.0)
        self.assertEqual(metrics["slot_recall"], 0.0)
        self.assertEqual(metrics["slot_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
