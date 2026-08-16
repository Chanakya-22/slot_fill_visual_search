import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.dataset import MultiModalDataset
from src.models.baseline_text import BaselineTextModel
from src.models.baseline_concat import BaselineConcatModel
from src.models.cross_attention_model import CrossAttentionModel
from src.utils import compute_intent_metrics, compute_slot_metrics

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BATCH_SIZE = 8
LR = 3e-5
LAMBDA_SLOT = 1.0  # Joint loss balance parameter

def train_single_model(model_name, model, train_loader, val_loader):
    print(f"\n==========================================")
    print(f"   Training: {model_name}")
    print(f"==========================================")
    
    model = model.to(DEVICE)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
    
    # Loss functions
    intent_criterion = nn.CrossEntropyLoss()
    slot_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            intent_labels = batch["intent_label"].to(DEVICE)
            slot_labels = batch["slot_labels"].to(DEVICE)

            if model_name == "BaselineTextModel":
                intent_logits, slot_logits = model(input_ids, attention_mask)
            else:
                pixel_values = batch["pixel_values"].to(DEVICE)
                intent_logits, slot_logits = model(input_ids, attention_mask, pixel_values)

            # Compute Multi-Task Loss: L_total = L_intent + lambda * L_slots
            loss_intent = intent_criterion(intent_logits, intent_labels)
            loss_slot = slot_criterion(slot_logits.view(-1, slot_logits.shape[-1]), slot_labels.view(-1))
            total_loss = loss_intent + (LAMBDA_SLOT * loss_slot)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        all_intent_preds, all_intent_targets = [], []
        all_slot_preds, all_slot_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                intent_labels = batch["intent_label"].to(DEVICE)
                slot_labels = batch["slot_labels"].to(DEVICE)

                if model_name == "BaselineTextModel":
                    intent_logits, slot_logits = model(input_ids, attention_mask)
                else:
                    pixel_values = batch["pixel_values"].to(DEVICE)
                    intent_logits, slot_logits = model(input_ids, attention_mask, pixel_values)

                # Intent predictions
                intent_preds = torch.argmax(intent_logits, dim=-1).cpu().numpy()
                all_intent_preds.extend(intent_preds)
                all_intent_targets.extend(intent_labels.cpu().numpy())

                # Slot predictions
                slot_preds = torch.argmax(slot_logits, dim=-1).cpu().numpy()
                all_slot_preds.extend(slot_preds)
                all_slot_targets.extend(slot_labels.cpu().numpy())

        intent_metrics = compute_intent_metrics(all_intent_preds, all_intent_targets)
        slot_metrics = compute_slot_metrics(all_slot_preds, all_slot_targets)

        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Intent Acc: {intent_metrics['accuracy']*100:.2f}% | "
              f"Val Slot F1: {slot_metrics['slot_f1']:.4f}")

    return {
        "final_intent_acc": intent_metrics["accuracy"] * 100,
        "final_slot_f1": slot_metrics["slot_f1"],
        "model": model
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "raw", "dataset.csv")

    dataset = MultiModalDataset(csv_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loaded {len(dataset)} samples. (Train: {train_size}, Val: {val_size}) using device: {DEVICE}")

    # Benchmark All Three Models
    models_to_benchmark = [
        ("BaselineTextModel", BaselineTextModel(num_intents=5, num_slots=4)),
        ("BaselineConcatModel", BaselineConcatModel(num_intents=5, num_slots=4)),
        ("CrossAttentionModel (Proposed)", CrossAttentionModel(num_intents=5, num_slots=4))
    ]

    results = []
    trained_novel_model = None

    for name, model_instance in models_to_benchmark:
        res = train_single_model(name, model_instance, train_loader, val_loader)
        results.append((name, res["final_intent_acc"], res["final_slot_f1"]))
        if "CrossAttention" in name:
            trained_novel_model = res["model"]

    # Print Comparison Table
    print("\n" + "="*60)
    print("           MID-REVIEW COMPARATIVE EVALUATION TABLE           ")
    print("="*60)
    print(f"{'Model Architecture':<32} | {'Intent Acc (%)':<15} | {'Slot F1':<10}")
    print("-" * 60)
    for name, acc, f1 in results:
        print(f"{name:<32} | {acc:<15.2f} | {f1:<10.4f}")
    print("="*60)

    # Save trained proposed model weights
    os.makedirs(os.path.join(base_dir, "data", "processed"), exist_ok=True)
    save_path = os.path.join(base_dir, "data", "processed", "cross_attention_model.pt")
    torch.save(trained_novel_model.state_dict(), save_path)
    print(f"\nTrained novel cross-attention model checkpoint saved to '{save_path}'.")

if __name__ == "__main__":
    main()