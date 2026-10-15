"""Create comparison tables, plots, and error summaries from experiment JSON."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _completed_models(results):
    return [
        model for model in results.get("models", [])
        if model.get("status") == "completed"
    ]


def _comparison_row(model):
    test = model["test_metrics"]
    intent = test["intent"]
    slots = test["slots"]
    latency = model.get("inference_latency") or {}
    parameters = model.get("parameter_counts") or {}
    ablation = model.get("image_pairing_ablation") or {}
    ablation_metrics = ablation.get("metrics", {})
    ablation_intent = ablation_metrics.get("intent", {})
    ablation_slots = ablation_metrics.get("slots", {})
    return {
        "model_name": model["model_name"],
        "status": model["status"],
        "split": model.get("dataset_split"),
        "seed": model.get("seed"),
        "selected_epoch": model.get("best_epoch"),
        "intent_accuracy": intent.get("accuracy"),
        "intent_macro_f1": intent.get("macro_f1"),
        "slot_entity_precision": slots.get("slot_precision"),
        "slot_entity_recall": slots.get("slot_recall"),
        "slot_entity_f1": slots.get("slot_f1"),
        "training_time_seconds": model.get("training_time_seconds"),
        "batch_size_one_latency_ms": latency.get("mean_batch_latency_ms"),
        "parameters_total": parameters.get("total"),
        "parameters_trainable": parameters.get("trainable"),
        "image_ablation_status": ablation.get("status"),
        "zero_image_intent_accuracy": ablation_intent.get("accuracy"),
        "zero_image_intent_macro_f1": ablation_intent.get("macro_f1"),
        "zero_image_slot_entity_f1": ablation_slots.get("slot_f1"),
    }


def _plot_comparison(rows, path):
    columns = (
        ("Intent accuracy", "intent_accuracy"),
        ("Intent macro F1", "intent_macro_f1"),
        ("Slot entity F1", "slot_entity_f1"),
    )
    names = [row["model_name"] for row in rows]
    x = np.arange(len(names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(max(9, len(names) * 3), 5))
    for index, (label, key) in enumerate(columns):
        values = [row[key] if row[key] is not None else 0.0 for row in rows]
        bars = axis.bar(x + (index - 1) * width, values, width, label=label)
        for bar, row in zip(bars, rows):
            if row[key] is None:
                bar.set_alpha(0.15)
    axis.set_xticks(x, names, rotation=12, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Held-out test-set metrics (completed models only)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_learning_curves(models, path):
    figure, (loss_axis, score_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in models:
        history = model.get("epoch_history", [])
        if not history:
            continue
        epochs = [entry["epoch"] for entry in history]
        losses = [entry["train_loss"] for entry in history]
        slot_f1 = [
            entry["validation_metrics"]["slots"]["slot_f1"]
            for entry in history
        ]
        loss_axis.plot(epochs, losses, marker="o", label=model["model_name"])
        score_axis.plot(epochs, slot_f1, marker="o", label=model["model_name"])
    loss_axis.set_title("Training loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    score_axis.set_title("Validation entity-level slot F1")
    score_axis.set_xlabel("Epoch")
    score_axis.set_ylabel("Micro F1")
    for axis in (loss_axis, score_axis):
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_confusion_matrices(models, intent_names, output_dir):
    paths = []
    for model in models:
        matrix = np.asarray(
            model["test_metrics"]["intent_confusion_matrix"], dtype=int
        )
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=range(len(intent_names)),
            yticks=range(len(intent_names)),
            xticklabels=intent_names,
            yticklabels=intent_names,
            xlabel="Predicted intent",
            ylabel="True intent",
            title=f"Test intent confusion: {model['model_name']}",
        )
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
        threshold = matrix.max() / 2 if matrix.size else 0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="white"
                    if matrix[row_index, column_index] > threshold
                    else "black",
                )
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        path = output_dir / f"{model['model_name']}_confusion_matrix.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def _error_analysis(models, output_dir):
    analysis = {}
    for model in models:
        examples = model["test_metrics"].get("examples", [])
        intent_errors = Counter()
        slot_errors = Counter()
        for example in examples:
            true_intent = example["intent_true"]
            predicted_intent = example["intent_predicted"]
            if not example["intent_correct"]:
                intent_errors[f"{true_intent} -> {predicted_intent}"] += 1
            for true_tag, predicted_tag in zip(
                example["slot_true_tags"], example["slot_predicted_tags"]
            ):
                if true_tag != predicted_tag:
                    slot_errors[f"{true_tag} -> {predicted_tag}"] += 1
        analysis[model["model_name"]] = {
            "test_example_count": len(examples),
            "intent_error_count": sum(intent_errors.values()),
            "intent_error_transitions": dict(intent_errors.most_common()),
            "slot_mismatched_token_count": sum(slot_errors.values()),
            "slot_tag_error_transitions": dict(slot_errors.most_common()),
            "representative_correct": [
                example
                for example in examples
                if example["intent_correct"] and example["slots_correct"]
            ][:5],
            "representative_incorrect": [
                example
                for example in examples
                if not example["intent_correct"] or not example["slots_correct"]
            ][:10],
        }
    _write_json(output_dir / "error_analysis.json", analysis)
    return analysis


def generate_results(experiment_path):
    experiment_path = Path(experiment_path).resolve()
    results = json.loads(experiment_path.read_text(encoding="utf-8"))
    output_dir = experiment_path.parent / f"{experiment_path.stem}_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    models = _completed_models(results)
    rows = [_comparison_row(model) for model in models]
    csv_path = output_dir / "model_comparison.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    artifacts = {
        "experiment_status": {
            model["model_name"]: model["status"]
            for model in results.get("models", [])
        },
        "completed_model_count": len(models),
        "comparison_csv": str(csv_path) if rows else None,
    }
    if models:
        comparison_plot = output_dir / "model_comparison.png"
        learning_curve = output_dir / "learning_curves.png"
        _plot_comparison(rows, comparison_plot)
        _plot_learning_curves(models, learning_curve)
        confusion_paths = _plot_confusion_matrices(
            models,
            results["intent_labels"],
            output_dir,
        )
        _error_analysis(models, output_dir)
        artifacts.update(
            {
                "model_comparison_plot": str(comparison_plot),
                "learning_curves_plot": str(learning_curve),
                "intent_confusion_matrices": [str(path) for path in confusion_paths],
                "error_analysis_json": str(output_dir / "error_analysis.json"),
            }
        )
    _write_json(output_dir / "artifact_manifest.json", artifacts)
    return output_dir, artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_json", type=Path)
    args = parser.parse_args(argv)
    output_dir, artifacts = generate_results(args.experiment_json)
    print(f"Results artifacts saved to {output_dir}")
    print(json.dumps(artifacts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
