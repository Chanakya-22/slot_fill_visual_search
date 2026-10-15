from sklearn.metrics import accuracy_score, f1_score

from src.labels import SLOT_ID_TO_TAG


def compute_intent_metrics(preds, labels):
    """
    Computes Accuracy and Macro F1 for Intent Classification.
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1": f1}

def compute_slot_metrics(preds, labels, label_map=None):
    """
    Computes token accuracy and exact BIO entity-level micro F1,
    filtering out special tokens and padding (-100).
    """
    tag_by_id = label_map or SLOT_ID_TO_TAG
    true_sequences = []
    pred_sequences = []

    for p_seq, l_seq in zip(preds, labels):
        if len(p_seq) != len(l_seq):
            raise ValueError("Slot prediction and target sequences must have equal lengths.")
        true_sequence = []
        pred_sequence = []
        for p, l in zip(p_seq, l_seq):
            if l != -100:
                if int(p) not in tag_by_id:
                    raise ValueError(f"Predicted slot ID {p} is not in the slot label map.")
                if int(l) not in tag_by_id:
                    raise ValueError(f"Target slot ID {l} is not in the slot label map.")
                true_sequence.append(tag_by_id[int(l)])
                pred_sequence.append(tag_by_id[int(p)])
        if true_sequence:
            true_sequences.append(true_sequence)
            pred_sequences.append(pred_sequence)
    if len(preds) != len(labels):
        raise ValueError("Slot prediction and target batches must have equal lengths.")

    if not true_sequences:
        return {"slot_accuracy": 0.0, "slot_f1": 0.0}

    flattened_true = [tag for sequence in true_sequences for tag in sequence]
    flattened_pred = [tag for sequence in pred_sequences for tag in sequence]
    accuracy = accuracy_score(flattened_true, flattened_pred)

    def extract_entities(sequence, sequence_index):
        entities = set()
        active_type = None
        active_start = None
        for token_index, tag in enumerate(sequence + ["O"]):
            if tag == "O":
                prefix, entity_type = "O", None
            else:
                prefix, entity_type = tag.split("-", 1)
            if active_type is not None and (prefix != "I" or entity_type != active_type):
                entities.add((sequence_index, active_start, token_index, active_type))
                active_type = None
                active_start = None
            if prefix == "B" or (prefix == "I" and active_type is None):
                active_type = entity_type
                active_start = token_index
        return entities

    true_entities = set()
    predicted_entities = set()
    for sequence_index, (true_sequence, pred_sequence) in enumerate(
        zip(true_sequences, pred_sequences)
    ):
        true_entities.update(extract_entities(true_sequence, sequence_index))
        predicted_entities.update(extract_entities(pred_sequence, sequence_index))

    true_positive = len(true_entities & predicted_entities)
    precision = true_positive / len(predicted_entities) if predicted_entities else 0.0
    recall = true_positive / len(true_entities) if true_entities else 0.0
    entity_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"slot_accuracy": accuracy, "slot_f1": entity_f1}