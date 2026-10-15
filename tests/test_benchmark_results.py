import json
import tempfile
import unittest
from pathlib import Path

from benchmark_results import generate_results


class BenchmarkResultTests(unittest.TestCase):
    def test_completed_results_generate_machine_readable_tables_and_plots(self):
        example = {
            "sample_id": "sample_1",
            "product_id": "product_0001",
            "text_query": "find gear",
            "intent_true_id": 0,
            "intent_true": "search_product",
            "intent_predicted_id": 0,
            "intent_predicted": "search_product",
            "intent_correct": True,
            "slot_tokens": ["find", "gear"],
            "slot_true_tags": ["O", "B-CATEGORY"],
            "slot_predicted_tags": ["O", "B-CATEGORY"],
            "slots_correct": True,
        }
        with tempfile.TemporaryDirectory(prefix="benchmark-results-test-") as temp_dir:
            experiment_path = Path(temp_dir) / "experiment.json"
            experiment_path.write_text(
                json.dumps(
                    {
                        "intent_labels": ["search_product", "other"],
                        "models": [
                            {
                                "status": "completed",
                                "model_name": "SmokeModel",
                                "dataset_split": "product",
                                "seed": 42,
                                "best_epoch": 1,
                                "test_metrics": {
                                    "intent": {
                                        "accuracy": 1.0,
                                        "macro_f1": 0.5,
                                        "class_metrics": {},
                                    },
                                    "slots": {
                                        "slot_precision": 1.0,
                                        "slot_recall": 1.0,
                                        "slot_f1": 1.0,
                                    },
                                    "intent_confusion_matrix": [[1, 0], [0, 0]],
                                    "examples": [example],
                                },
                                "inference_latency": {
                                    "mean_batch_latency_ms": 1.25
                                },
                                "parameter_counts": {
                                    "total": 100,
                                    "trainable": 50,
                                },
                                "training_time_seconds": 10.0,
                                "image_pairing_ablation": {
                                    "status": "not_applicable"
                                },
                                "epoch_history": [
                                    {
                                        "epoch": 1,
                                        "train_loss": 1.0,
                                        "validation_metrics": {
                                            "slots": {"slot_f1": 0.5}
                                        },
                                    }
                                ],
                            },
                            {"model_name": "IncompleteModel", "status": "in_progress"},
                        ],
                        "dataset_provenance": {
                            "dataset_summary": {
                                "intent_distribution": {}
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            output_dir, manifest = generate_results(experiment_path)
            self.assertEqual(manifest["completed_model_count"], 1)
            self.assertEqual(
                manifest["experiment_status"]["IncompleteModel"], "in_progress"
            )
            for name in (
                "model_comparison.csv",
                "model_comparison.png",
                "learning_curves.png",
                "SmokeModel_confusion_matrix.png",
                "error_analysis.json",
                "artifact_manifest.json",
            ):
                self.assertTrue((output_dir / name).is_file(), name)
            error_analysis = json.loads(
                (output_dir / "error_analysis.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                error_analysis["SmokeModel"]["representative_correct"][0]["sample_id"],
                "sample_1",
            )
            comparison = (output_dir / "model_comparison.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("slot_entity_f1", comparison)
            self.assertNotIn("IncompleteModel", comparison)


if __name__ == "__main__":
    unittest.main()
