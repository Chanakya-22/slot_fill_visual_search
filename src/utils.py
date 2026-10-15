from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.labels import INTENT_ID_TO_NAME, SLOT_ID_TO_TAG


def compute_intent_metrics(preds, labels):
    """Return accuracy, macro F1, and per-intent precision/recall/F1."""
    if len(preds) != len(labels):
        raise ValueError("Intent prediction and target batches must have equal lengths.")
    if len(labels) == 0:
        raise ValueError("Intent metrics require at least one example.")

    label_ids = list(INTENT_ID_TO_NAME)
    for value in [*preds, *labels]:
        if int(value) not in INTENT_ID_TO_NAME:
            raise ValueError(f"Intent ID {value} is not in the intent label map.")

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=label_ids,
        zero_division=0,
    )
    class_metrics = {
        INTENT_ID_TO_NAME[label_id]: {
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "support": int(class_support),
        }
        for label_id, class_precision, class_recall, class_f1, class_support in zip(
            label_ids, precision, recall, f1, support
        )
    }
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1.mean()),
        "class_metrics": class_metrics,
    }


def _entities_from_tags(tags, sequence_index, start_offset=0):
    entities = set()
    active_type = None
    active_start = None
    for token_index, tag in enumerate([*tags, "O"]):
        if tag == "O":
            prefix, entity_type = "O", None
        else:
            prefix, entity_type = tag.split("-", 1)

        if active_type is not None and (prefix != "I" or entity_type != active_type):
            entities.add(
                (sequence_index, active_start, start_offset + token_index, active_type)
            )
            active_type = None
            active_start = None

        if prefix == "B" or (prefix == "I" and active_type is None):
            active_type = entity_type
            active_start = start_offset + token_index

    return entities


def _split_unmasked_sequences(prediction, target, tag_by_id):
    true_sequences = []
    predicted_sequences = []
    true_segment = []
    predicted_segment = []

    for predicted_id, target_id in zip(prediction, target):
        if int(target_id) == -100:
            if true_segment:
                true_sequences.append(true_segment)
                predicted_sequences.append(predicted_segment)
                true_segment = []
                predicted_segment = []
            continue

        predicted_id = int(predicted_id)
        target_id = int(target_id)
        if predicted_id not in tag_by_id:
            raise ValueError(f"Predicted slot ID {predicted_id} is not in the slot label map.")
        if target_id not in tag_by_id:
            raise ValueError(f"Target slot ID {target_id} is not in the slot label map.")
        predicted_segment.append(tag_by_id[predicted_id])
        true_segment.append(tag_by_id[target_id])

    if true_segment:
        true_sequences.append(true_segment)
        predicted_sequences.append(predicted_segment)
    return true_sequences, predicted_sequences


def compute_slot_metrics(preds, labels, label_map=None):
    """Compute exact-span/type micro scores over BIO entities.

    Positions with target ``-100`` (special tokens, padding, or ignored
    subwords) end a segment and are excluded from both entity extraction and
    scoring. A malformed ``I-`` tag without a matching active entity starts a
    new entity, matching the common permissive BIO evaluation convention.
    """
    tag_by_id = label_map or SLOT_ID_TO_TAG
    if len(preds) != len(labels):
        raise ValueError("Slot prediction and target batches must have equal lengths.")

    true_entities = set()
    predicted_entities = set()
    sequence_index = 0
    for prediction, target in zip(preds, labels):
        if len(prediction) != len(target):
            raise ValueError("Slot prediction and target sequences must have equal lengths.")

        true_sequences, predicted_sequences = _split_unmasked_sequences(
            prediction, target, tag_by_id
        )
        for true_sequence, predicted_sequence in zip(
            true_sequences, predicted_sequences
        ):
            true_entities.update(
                _entities_from_tags(true_sequence, sequence_index)
            )
            predicted_entities.update(
                _entities_from_tags(predicted_sequence, sequence_index)
            )
            sequence_index += 1

    true_positive = len(true_entities & predicted_entities)
    precision = true_positive / len(predicted_entities) if predicted_entities else 0.0
    recall = true_positive / len(true_entities) if true_entities else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "slot_precision": precision,
        "slot_recall": recall,
        "slot_f1": f1,
        "true_entity_count": len(true_entities),
        "predicted_entity_count": len(predicted_entities),
    }
