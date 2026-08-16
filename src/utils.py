import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_intent_metrics(preds, labels):
    """
    Computes Accuracy and Macro F1 for Intent Classification.
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "f1": f1}

def compute_slot_metrics(preds, labels, label_map=None):
    """
    Computes token-level accuracy and F1 score for Slot Tagging,
    filtering out special tokens and padding (-100).
    """
    true_labels = []
    pred_labels = []

    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l != -100:  # Ignore padded/special positions
                true_labels.append(l)
                pred_labels.append(p)

    if not true_labels:
        return {"slot_accuracy": 0.0, "slot_f1": 0.0}

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    return {"slot_accuracy": acc, "slot_f1": f1}