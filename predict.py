import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from src.models.cross_attention_model import CrossAttentionModel
from src.labels import INTENT_ID_TO_NAME, NUM_INTENTS, NUM_SLOT_LABELS, SLOT_ID_TO_TAG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTENT_MAP = INTENT_ID_TO_NAME
SLOT_MAP = SLOT_ID_TO_TAG

def predict(image_path, text_query, model_path="data/processed/cross_attention_model.pt"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint = os.path.join(base_dir, model_path)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load Model
    model = CrossAttentionModel(num_intents=NUM_INTENTS, num_slots=NUM_SLOT_LABELS)
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        print("Loaded trained model weights successfully.")
    else:
        print("Warning: Checkpoint not found. Using initialized model weights.")

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
    for token, slot_id in zip(tokens, pred_slot_ids):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        tag = SLOT_MAP.get(slot_id, "O")
        if tag != "O":
            extracted_slots.append(f"{token} -> {tag}")

    print("\n--- INFERENCE PREDICTION RESULTS ---")
    print(f"Input Query:      '{text_query}'")
    print(f"Target Image:     '{image_path}'")
    print(f"Predicted Intent: {INTENT_MAP.get(pred_intent_id, 'unknown')} (ID: {pred_intent_id})")
    print(f"Extracted Slots:  {extracted_slots if extracted_slots else 'None (all tokens tagged as O)'}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_img = os.path.join(base_dir, "data", "raw", "images", "sample_000.jpg")
    sample_text = "find this gear in titanium"
    predict(sample_img, sample_text)