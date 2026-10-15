import os
import torch
from src.dataset import MultiModalDataset
from src.models.baseline_text import BaselineTextModel
from src.models.baseline_concat import BaselineConcatModel
from src.models.cross_attention_model import CrossAttentionModel
from src.labels import NUM_INTENTS, NUM_SLOT_LABELS

def run_sanity_check():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "raw", "dataset.csv")

    # 1. Ensure mock data exists
    if not os.path.exists(csv_path):
        print("Synthetic data not found. Running prepare_hardware_data.py first...")
        from prepare_hardware_data import compile_hardware_corpus
        compile_hardware_corpus()

    # 2. Load dataset batch
    print("\n--- Loading Dataset Batch ---")
    dataset = MultiModalDataset(csv_path)
    sample = dataset[0]

    input_ids = sample["input_ids"].unsqueeze(0)        # Shape: [1, 32]
    attention_mask = sample["attention_mask"].unsqueeze(0)  # Shape: [1, 32]
    pixel_values = sample["pixel_values"].unsqueeze(0)    # Shape: [1, 3, 224, 224]

    print(f"Input IDs Shape: {input_ids.shape}")
    print(f"Pixel Values Shape: {pixel_values.shape}")

    # 3. Test Baseline 1: Text-Only Model
    print("\n--- Testing Baseline 1 (Text Model) ---")
    text_model = BaselineTextModel(num_intents=NUM_INTENTS, num_slots=NUM_SLOT_LABELS)
    intent_logits, slot_logits = text_model(input_ids, attention_mask)
    print(f"Intent Logits Output Shape: {intent_logits.shape} (Expected: [1, {NUM_INTENTS}])")
    print(f"Slot Logits Output Shape:   {slot_logits.shape}   (Expected: [1, 32, {NUM_SLOT_LABELS}])")

    # 4. Test Baseline 2: Early Fusion Concat Model
    print("\n--- Testing Baseline 2 (Concat Model) ---")
    concat_model = BaselineConcatModel(num_intents=NUM_INTENTS, num_slots=NUM_SLOT_LABELS)
    intent_logits_c, slot_logits_c = concat_model(input_ids, attention_mask, pixel_values)
    print(f"Concat Intent Logits Shape: {intent_logits_c.shape} (Expected: [1, {NUM_INTENTS}])")
    print(f"Concat Slot Logits Shape:   {slot_logits_c.shape}   (Expected: [1, 32, {NUM_SLOT_LABELS}])")
    
    # 5. Test Phase 3: Cross-Attention Model
    print("\n--- Testing Phase 3 (Cross-Attention Model) ---")
    cross_model = CrossAttentionModel(num_intents=NUM_INTENTS, num_slots=NUM_SLOT_LABELS)
    intent_logits_cross, slot_logits_cross = cross_model(input_ids, attention_mask, pixel_values)
    print(f"Cross-Attn Intent Logits Shape: {intent_logits_cross.shape} (Expected: [1, {NUM_INTENTS}])")
    print(f"Cross-Attn Slot Logits Shape:   {slot_logits_cross.shape}   (Expected: [1, 32, {NUM_SLOT_LABELS}])")

    print("\nSanity Check Passed Successfully!")
    
    

if __name__ == "__main__":
    run_sanity_check()