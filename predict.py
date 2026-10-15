import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from src.models.cross_attention_model import CrossAttentionModel
from src.labels import INTENT_ID_TO_NAME, NUM_INTENTS, NUM_SLOT_LABELS, SLOT_ID_TO_TAG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTENT_MAP = INTENT_ID_TO_NAME
SLOT_MAP = SLOT_ID_TO_TAG

def predict(
    image_path,
    text_query,
    model_path="data/processed/crossattentionmodel_product_seed42.pt",
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint = os.path.join(base_dir, model_path)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model = CrossAttentionModel(num_intents=NUM_INTENTS, num_slots=NUM_SLOT_LABELS)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Trained model checkpoint not found: {checkpoint}")
    checkpoint_data = torch.load(checkpoint, map_location=DEVICE)
    state_dict = (
        checkpoint_data["model_state_dict"]
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data
        else checkpoint_data
    )
    model.load_state_dict(state_dict)
    print("Loaded trained model weights successfully.")

    model.to(DEVICE)
    model.eval()

    # Preprocess inputs
    text_enc = tokenizer(text_query, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
    img = Image.open(image_path).convert("RGB")
    img_enc = processor(images=img, return_tensors="pt")

    input_ids = text_enc["input_ids"].to(DEVICE)
    attention_mask = text_enc["attention_mask"].to(DEVICE)
    pixel_values = img_enc["pixel_values"].to(DEVICE)

    with torch.no_grad():
        intent_logits, slot_logits = model(input_ids, attention_mask, pixel_values)

    pred_intent_id = torch.argmax(intent_logits, dim=-1).item()
    pred_slot_ids = torch.argmax(slot_logits, dim=-1).squeeze(0).cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    extracted_slots = []
    active_tokens = attention_mask.squeeze(0).cpu().numpy()
    for token, slot_id, is_active in zip(tokens, pred_slot_ids, active_tokens):
        if not is_active or token in tokenizer.all_special_tokens:
            continue
        tag = SLOT_MAP.get(slot_id, "O")
        if tag != "O":
            extracted_slots.append(f"{token} -> {tag}")

    print("\n--- INFERENCE PREDICTION RESULTS ---")
    print(f"Input Query:      '{text_query}'")
    print(f"Target Image:     '{image_path}'")
    print(f"Predicted Intent: {INTENT_MAP.get(pred_intent_id, 'unknown')} (ID: {pred_intent_id})")
    print(f"Extracted Slots:  {extracted_slots if extracted_slots else 'None (all tokens tagged as O)'}")
    return {
        "intent_id": pred_intent_id,
        "intent": INTENT_MAP.get(pred_intent_id, "unknown"),
        "slot_predictions": extracted_slots,
    }

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_img = os.path.join(base_dir, "data", "raw", "images", "product_0000.png")
    sample_text = "find a forest green spiral tooth gear"
    predict(sample_img, sample_text)