"""Canonical intent and BIO slot label definitions used by the data pipeline."""

INTENT_ID_TO_NAME = {
    0: "search_product",
    1: "troubleshoot_product",
    2: "identify_product",
    3: "check_compatibility",
    4: "request_specifications",
}
INTENT_NAME_TO_ID = {name: intent_id for intent_id, name in INTENT_ID_TO_NAME.items()}
NUM_INTENTS = len(INTENT_ID_TO_NAME)

SLOT_TAGS = (
    "O",
    "B-CATEGORY",
    "I-CATEGORY",
    "B-COLOR",
    "I-COLOR",
    "B-SHAPE",
    "I-SHAPE",
    "B-MATERIAL",
    "I-MATERIAL",
    "B-PRICE",
    "I-PRICE",
)
SLOT_TAG_TO_ID = {tag: slot_id for slot_id, tag in enumerate(SLOT_TAGS)}
SLOT_ID_TO_TAG = {slot_id: tag for tag, slot_id in SLOT_TAG_TO_ID.items()}
NUM_SLOT_LABELS = len(SLOT_TAGS)
