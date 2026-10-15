"""Train and evaluate intent/slot models on reproducible grouped splits."""

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from src.dataset import MultiModalDataset
from src.labels import (
    INTENT_ID_TO_NAME,
    NUM_INTENTS,
    NUM_SLOT_LABELS,
    SLOT_ID_TO_TAG,
)
from src.models.baseline_concat import BaselineConcatModel
from src.models.baseline_text import BaselineTextModel
from src.models.cross_attention_model import CrossAttentionModel
from src.utils import compute_intent_metrics, compute_slot_metrics

SEED = 42
BATCH_SIZE = 8
LR = 3e-5
LAMBDA_SLOT = 1.0
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
TEMPLATE_VALIDATION_FRACTION = 0.25
TEMPLATE_TEST_FRACTION = 0.25
MODEL_TYPES = {
    "BaselineTextModel": BaselineTextModel,
    "BaselineConcatModel": BaselineConcatModel,
    "CrossAttentionModel": CrossAttentionModel,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _group_values(dataset, split_mode):
    if "intent_id" not in dataset.df:
        raise ValueError("Dataset must provide intent_id values for stratified splits.")
    intent_ids = dataset.df["intent_id"].astype(int).to_numpy()
    if set(intent_ids) != set(INTENT_ID_TO_NAME):
        raise ValueError(
            "Every configured intent must occur in the dataset before splitting."
        )

    if split_mode == "product":
        if "product_id" not in dataset.df or dataset.df["product_id"].isna().any():
            raise ValueError(
                "Product-generalization evaluation requires product_id on every row."
            )
        groups = dataset.df["product_id"].astype(str).to_numpy()
    elif split_mode == "template":
        if "generation_info" not in dataset.df:
            raise ValueError(
                "Template-generalization evaluation requires generation_info."
            )
        template_ids = []
        for row_index, value in enumerate(dataset.df["generation_info"]):
            try:
                generation_info = json.loads(value) if isinstance(value, str) else value
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"Invalid generation_info at row {row_index}."
                ) from exc
            template_id = generation_info.get("template_id") if isinstance(
                generation_info, dict
            ) else None
            if not isinstance(template_id, str) or not template_id:
                raise ValueError(
                    f"Missing query template ID at row {row_index}."
                )
            template_ids.append(template_id)
        groups = np.asarray(template_ids)
    else:
        raise ValueError("split_mode must be 'product' or 'template'.")
    if len(set(groups)) < 3:
        raise ValueError("At least three distinct groups are needed for three splits.")
    return groups, intent_ids


def split_dataset(
    dataset,
    validation_fraction=None,
    test_fraction=None,
    seed=SEED,
    split_mode="product",
):
    """Create deterministic, mutually exclusive grouped train/val/test subsets."""
    if validation_fraction is None:
        validation_fraction = (
            TEMPLATE_VALIDATION_FRACTION
            if split_mode == "template"
            else VALIDATION_FRACTION
        )
    if test_fraction is None:
        test_fraction = (
            TEMPLATE_TEST_FRACTION if split_mode == "template" else TEST_FRACTION
        )
    if (
        not 0 < validation_fraction < 1
        or not 0 < test_fraction < 1
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError("Validation and test fractions must be positive and sum below one.")

    groups, intent_ids = _group_values(dataset, split_mode)
    all_intents = set(intent_ids)
    split_indices = None
    last_error = None
    group_intents = {
        group: set(intent_ids[groups == group])
        for group in set(groups)
    }
    if all(len(intents) == 1 for intents in group_intents.values()):
        groups_by_intent = {
            intent_id: sorted(
                group
                for group, intents in group_intents.items()
                if next(iter(intents)) == intent_id
            )
            for intent_id in all_intents
        }
        if any(len(intent_groups) < 3 for intent_groups in groups_by_intent.values()):
            raise ValueError(
                "Pure-class groups need at least three groups per intent for "
                "train/validation/test template generalization."
            )
        rng = np.random.default_rng(seed)
        indices_by_split = {"train": [], "validation": [], "test": []}
        for intent_groups in groups_by_intent.values():
            shuffled_groups = list(rng.permutation(intent_groups))
            validation_count = max(1, round(len(intent_groups) * validation_fraction))
            test_count = max(1, round(len(intent_groups) * test_fraction))
            while validation_count + test_count >= len(intent_groups):
                if validation_count >= test_count:
                    validation_count -= 1
                else:
                    test_count -= 1
            validation_groups = shuffled_groups[:validation_count]
            test_groups = shuffled_groups[
                validation_count : validation_count + test_count
            ]
            train_groups = shuffled_groups[validation_count + test_count :]
            for split_name, split_groups in (
                ("train", train_groups),
                ("validation", validation_groups),
                ("test", test_groups),
            ):
                indices_by_split[split_name].extend(
                    np.flatnonzero(np.isin(groups, split_groups)).tolist()
                )
        split_indices = tuple(
            sorted(indices_by_split[name])
            for name in ("train", "validation", "test")
        )
    else:
        for attempt in range(256):
            split_seed = (int(seed) + attempt * 2) % (2**32 - 1)
            train_val_test_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=validation_fraction + test_fraction,
                random_state=split_seed,
            )
            try:
                train_indices, held_indices = next(
                    train_val_test_splitter.split(
                        np.zeros(len(groups)), intent_ids, groups
                    )
                )
                validation_test_splitter = GroupShuffleSplit(
                    n_splits=1,
                    test_size=test_fraction / (validation_fraction + test_fraction),
                    random_state=(split_seed + 1) % (2**32 - 1),
                )
                validation_relative, test_relative = next(
                    validation_test_splitter.split(
                        np.zeros(len(held_indices)),
                        intent_ids[held_indices],
                        groups[held_indices],
                    )
                )
            except ValueError as exc:
                last_error = exc
                break

            validation_indices = held_indices[validation_relative]
            test_indices = held_indices[test_relative]
            partitions = (train_indices, validation_indices, test_indices)
            if all(set(intent_ids[indices]) == all_intents for indices in partitions):
                split_indices = tuple(
                    np.sort(indices).astype(int).tolist() for indices in partitions
                )
                break

    if split_indices is None:
        reason = f" Split error: {last_error}" if last_error else ""
        raise ValueError(
            "Could not create three grouped splits containing every intent "
            f"after 256 deterministic attempts.{reason}"
        )

    train_indices, validation_indices, test_indices = split_indices
    subsets = tuple(
        Subset(dataset, indices)
        for indices in (train_indices, validation_indices, test_indices)
    )
    split_details = {
        "mode": split_mode,
        "seed": int(seed),
        "fractions": {
            "train": 1.0 - validation_fraction - test_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "partitions": {},
    }
    for name, indices in zip(
        ("train", "validation", "test"),
        (train_indices, validation_indices, test_indices),
    ):
        partition_groups = sorted(set(groups[indices]))
        counts = Counter(intent_ids[indices])
        sample_ids = (
            dataset.df.iloc[indices]["sample_id"].astype(str).tolist()
            if "sample_id" in dataset.df
            else [str(index) for index in indices]
        )
        split_details["partitions"][name] = {
            "example_count": len(indices),
            "group_count": len(partition_groups),
            "group_ids": partition_groups,
            "sample_ids": sample_ids,
            "intent_distribution": {
                INTENT_ID_TO_NAME[intent_id]: counts[intent_id]
                for intent_id in INTENT_ID_TO_NAME
            },
        }
    return (*subsets, split_details)


def _forward(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    if isinstance(model, BaselineTextModel):
        return model(input_ids, attention_mask)
    return model(input_ids, attention_mask, batch["pixel_values"].to(device))


def evaluate(model, data_loader, device, include_examples=False, zero_images=False):
    model.eval()
    intent_predictions = []
    intent_targets = []
    slot_predictions = []
    slot_targets = []
    sample_ids = []
    text_queries = []
    product_ids = []
    input_token_ids = []
    with torch.no_grad():
        for batch in data_loader:
            eval_batch = batch
            if zero_images:
                eval_batch = dict(batch)
                eval_batch["pixel_values"] = torch.zeros_like(batch["pixel_values"])
            intent_logits, slot_logits = _forward(model, eval_batch, device)
            intent_predictions.extend(intent_logits.argmax(dim=-1).cpu().tolist())
            intent_targets.extend(batch["intent_label"].cpu().tolist())
            slot_predictions.extend(slot_logits.argmax(dim=-1).cpu().tolist())
            slot_targets.extend(batch["slot_labels"].cpu().tolist())
            sample_ids.extend(batch.get("sample_id", []))
            text_queries.extend(batch.get("text_query", []))
            product_ids.extend(batch.get("product_id", []))
            input_token_ids.extend(batch["input_ids"].cpu().tolist())

    result = {
        "intent": compute_intent_metrics(intent_predictions, intent_targets),
        "slots": compute_slot_metrics(slot_predictions, slot_targets),
        "intent_confusion_matrix": confusion_matrix(
            intent_targets,
            intent_predictions,
            labels=list(INTENT_ID_TO_NAME),
        ).tolist(),
    }
    if include_examples and sample_ids:
        dataset = data_loader.dataset
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        tokenizer = getattr(dataset, "tokenizer", None)
        examples = []
        for index, (sample_id, text_query, product_id) in enumerate(
            zip(sample_ids, text_queries, product_ids)
        ):
            target_slot_ids = slot_targets[index]
            predicted_slot_ids = slot_predictions[index]
            active_positions = [
                position
                for position, target_id in enumerate(target_slot_ids)
                if target_id != -100
            ]
            tokens = (
                tokenizer.convert_ids_to_tokens(
                    [input_token_ids[index][position] for position in active_positions]
                )
                if tokenizer is not None
                else [str(position) for position in active_positions]
            )
            target_tags = [
                SLOT_ID_TO_TAG[target_slot_ids[position]]
                for position in active_positions
            ]
            predicted_tags = [
                SLOT_ID_TO_TAG[predicted_slot_ids[position]]
                for position in active_positions
            ]
            examples.append(
                {
                    "sample_id": sample_id,
                    "product_id": product_id,
                    "text_query": text_query,
                    "intent_true_id": int(intent_targets[index]),
                    "intent_true": INTENT_ID_TO_NAME[int(intent_targets[index])],
                    "intent_predicted_id": int(intent_predictions[index]),
                    "intent_predicted": INTENT_ID_TO_NAME[
                        int(intent_predictions[index])
                    ],
                    "intent_correct": int(intent_targets[index])
                    == int(intent_predictions[index]),
                    "slot_tokens": tokens,
                    "slot_true_tags": target_tags,
                    "slot_predicted_tags": predicted_tags,
                    "slots_correct": target_tags == predicted_tags,
                }
            )
        result["examples"] = examples
    return result


def _dataset_provenance(csv_path):
    summary_path = csv_path.parent / "dataset_summary.json"
    catalogue_path = csv_path.parent / "catalogue.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else None
    )
    return {
        "dataset_version": (summary or {}).get("image_generation_method", "unknown"),
        "dataset_summary": summary,
        "dataset_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "catalogue_sha256": hashlib.sha256(catalogue_path.read_bytes()).hexdigest()
        if catalogue_path.is_file()
        else None,
    }


def _parameter_counts(model):
    counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "backbones": {},
    }
    for name in ("bert", "clip_vision"):
        backbone = getattr(model, name, None)
        if backbone is None:
            counts["backbones"][name] = {"present": False}
            continue
        counts["backbones"][name] = {
            "present": True,
            "total": sum(parameter.numel() for parameter in backbone.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in backbone.parameters()
                if parameter.requires_grad
            ),
        }
    return counts


def _image_pairing_ablation(model, test_loader, device, model_name):
    if model_name == "BaselineTextModel":
        return {"status": "not_applicable", "reason": "text-only architecture"}
    metrics = evaluate(model, test_loader, device, zero_images=True)
    metrics.pop("examples", None)
    return {
        "status": "completed",
        "method": "replace normalized pixel_values with zeros at evaluation",
        "metrics": metrics,
    }


def measure_inference_latency(model, dataset, device, warmup_runs=2, timed_runs=5):
    """Measure batch-size-one latency on a real held-out example."""
    model.eval()
    sample = dataset[0]
    batch = {
        key: value.unsqueeze(0)
        for key, value in sample.items()
        if key in {"input_ids", "attention_mask", "pixel_values"}
    }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _forward(model, batch, device)
        synchronize()
        started = time.perf_counter()
        for _ in range(timed_runs):
            _forward(model, batch, device)
        synchronize()
    elapsed = time.perf_counter() - started
    return {
        "batch_size": 1,
        "mean_batch_latency_ms": elapsed * 1000.0 / timed_runs,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "device": str(device),
        "latency_scope": "model_forward_and_device_transfer; preprocessing excluded",
    }


def _save_checkpoint(path, state_dict, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": state_dict, "metadata": metadata},
        path,
    )


def train_single_model(
    model_name,
    model,
    train_loader,
    validation_loader,
    device,
    seed,
    split_mode,
    epochs,
    patience,
    checkpoint_path,
):
    model = model.to(device)
    optimizer = AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    intent_criterion = nn.CrossEntropyLoss()
    slot_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_slot_f1 = -1.0
    best_state = None
    best_epoch = 0
    best_validation_metrics = None
    epoch_history = []
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            intent_logits, slot_logits = _forward(model, batch, device)
            intent_labels = batch["intent_label"].to(device)
            slot_labels = batch["slot_labels"].to(device)
            intent_loss = intent_criterion(intent_logits, intent_labels)
            slot_loss = slot_criterion(
                slot_logits.reshape(-1, slot_logits.shape[-1]),
                slot_labels.reshape(-1),
            )
            loss = intent_loss + LAMBDA_SLOT * slot_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())

        validation_metrics = evaluate(model, validation_loader, device)
        validation_slot_f1 = validation_metrics["slots"]["slot_f1"]
        average_loss = total_loss / len(train_loader)
        print(
            f"{model_name} epoch {epoch}/{epochs}: "
            f"train_loss={average_loss:.4f} "
            f"val_intent_accuracy={validation_metrics['intent']['accuracy']:.4f} "
            f"val_intent_macro_f1={validation_metrics['intent']['macro_f1']:.4f} "
            f"val_slot_entity_f1={validation_slot_f1:.4f}"
        )
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": average_loss,
                "validation_metrics": validation_metrics,
            }
        )

        if validation_slot_f1 > best_slot_f1:
            best_slot_f1 = validation_slot_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_validation_metrics = validation_metrics
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_path,
                best_state,
                {
                    "model_name": model_name,
                    "seed": seed,
                    "split_mode": split_mode,
                    "best_epoch": best_epoch,
                    "checkpoint_selection_metric": "validation.slots.slot_f1",
                    "validation_metrics": best_validation_metrics,
                    "epoch_history": epoch_history,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping {model_name} at epoch {epoch}.")
                break

    training_time_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError(f"No validation checkpoint was produced for {model_name}.")
    model.load_state_dict(best_state)
    _save_checkpoint(
        checkpoint_path,
        best_state,
        {
            "model_name": model_name,
            "seed": seed,
            "split_mode": split_mode,
            "best_epoch": best_epoch,
            "training_time_seconds": training_time_seconds,
            "checkpoint_selection_metric": "validation.slots.slot_f1",
            "validation_metrics": best_validation_metrics,
            "epoch_history": epoch_history,
        },
    )
    return {
        "model": model,
        "best_epoch": best_epoch,
        "validation_metrics": best_validation_metrics,
        "epoch_history": epoch_history,
        "training_time_seconds": training_time_seconds,
        "checkpoint_path": checkpoint_path,
    }


def _make_model(model_name):
    model_type = MODEL_TYPES[model_name]
    return model_type(num_intents=NUM_INTENTS, num_slots=NUM_SLOT_LABELS)


def _run_experiment_model(
    model_name,
    train_set,
    validation_set,
    test_set,
    device,
    seed,
    split_mode,
    batch_size,
    epochs,
    patience,
    output_dir,
    base_dir,
):
    seed_everything(seed)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    checkpoint_path = output_dir / (
        f"{model_name.lower()}_{split_mode}_seed{seed}.pt"
    )
    model = _make_model(model_name)
    model_parameter_counts = _parameter_counts(model)
    trained = train_single_model(
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        seed=seed,
        split_mode=split_mode,
        epochs=epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
    )
    test_metrics = evaluate(
        trained["model"], test_loader, device, include_examples=True
    )
    ablation = _image_pairing_ablation(
        trained["model"], test_loader, device, model_name
    )
    latency = measure_inference_latency(trained["model"], test_set, device)
    model_result = {
        "status": "completed",
        "model_name": model_name,
        "dataset_split": split_mode,
        "seed": seed,
        "best_epoch": trained["best_epoch"],
        "checkpoint_selection_metric": "validation.slots.slot_f1",
        "validation_metrics": trained["validation_metrics"],
        "test_metrics": test_metrics,
        "image_pairing_ablation": ablation,
        "training_time_seconds": trained["training_time_seconds"],
        "inference_latency": latency,
        "parameter_counts": model_parameter_counts,
        "epoch_history": trained["epoch_history"],
        "checkpoint_path": os.path.relpath(checkpoint_path, base_dir),
    }
    del model, trained, train_loader, validation_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_result


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "raw" / "dataset.csv",
    )
    parser.add_argument(
        "--models",
        choices=("all", *MODEL_TYPES),
        default="all",
        help="Train all three models or one architecture for a quick smoke run.",
    )
    parser.add_argument("--split-mode", choices=("product", "template"), default="product")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("batch size, epochs, and patience must be positive.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    base_dir = Path(__file__).resolve().parent
    csv_path = args.csv.resolve()
    output_dir = (args.output_dir or base_dir / "data" / "processed").resolve()
    dataset = MultiModalDataset(csv_path)
    train_set, validation_set, test_set, split_details = split_dataset(
        dataset,
        seed=args.seed,
        split_mode=args.split_mode,
    )
    print(
        f"Device: {device}; grouped by {args.split_mode}; "
        f"train/validation/test sizes: "
        f"{len(train_set)}/{len(validation_set)}/{len(test_set)}."
    )

    model_names = list(MODEL_TYPES) if args.models == "all" else [args.models]
    results_path = output_dir / (
        f"experiment_results_{args.split_mode}_seed{args.seed}.json"
    )
    results = {
        "dataset_path": os.path.relpath(csv_path, base_dir),
        "seed": args.seed,
        "dataset_provenance": _dataset_provenance(csv_path),
        "dataset_split": split_details,
        "device": str(device),
        "runtime_environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "cpu": platform.processor(),
        },
        "training_configuration": {
            "batch_size": args.batch_size,
            "learning_rate": LR,
            "lambda_slot": LAMBDA_SLOT,
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "checkpoint_selection_metric": "validation.slots.slot_f1",
            "text_encoder": "bert-base-uncased",
            "vision_encoder": "openai/clip-vit-base-patch32",
            "cpu_threads": torch.get_num_threads()
            if device.type == "cpu"
            else None,
        },
        "intent_labels": [
            INTENT_ID_TO_NAME[intent_id]
            for intent_id in INTENT_ID_TO_NAME
        ],
        "models": [
            {"model_name": model_name, "status": "pending"}
            for model_name in model_names
        ],
    }
    _write_json(results_path, results)

    for model_name in model_names:
        model_result = next(
            result for result in results["models"]
            if result["model_name"] == model_name
        )
        model_result["status"] = "in_progress"
        _write_json(results_path, results)
        try:
            completed_result = _run_experiment_model(
                model_name=model_name,
                train_set=train_set,
                validation_set=validation_set,
                test_set=test_set,
                device=device,
                seed=args.seed,
                split_mode=args.split_mode,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                output_dir=output_dir,
                base_dir=base_dir,
            )
        except Exception as exc:
            model_result.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            _write_json(results_path, results)
            print(
                f"ERROR: {model_name} failed ({type(exc).__name__}): {exc}",
                flush=True,
            )
            raise
        model_result.clear()
        model_result.update(completed_result)
        _write_json(results_path, results)
        test_metrics = completed_result["test_metrics"]
        print(
            f"{model_name}: test_intent_accuracy="
            f"{test_metrics['intent']['accuracy']:.4f}, "
            f"test_intent_macro_f1={test_metrics['intent']['macro_f1']:.4f}, "
            f"test_slot_entity_f1={test_metrics['slots']['slot_f1']:.4f}, "
            f"training_time={completed_result['training_time_seconds']:.2f}s, "
            f"inference_latency="
            f"{completed_result['inference_latency']['mean_batch_latency_ms']:.2f}ms"
        )
    print(f"Experiment result state written to {results_path}")


if __name__ == "__main__":
    main()
