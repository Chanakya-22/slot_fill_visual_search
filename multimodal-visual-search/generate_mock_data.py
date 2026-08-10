from __future__ import annotations

from pathlib import Path
import random
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
IMAGE_DIR = DATA_DIR / "images"
CSV_PATH = DATA_DIR / "dataset.csv"

INTENT_LABELS = [
    "product_lookup",
    "product_comparison",
    "attribute_filter",
    "size_color_match",
    "brand_style_match",
]

COLOR_WORDS = {
    "black", "blue", "red", "green", "white", "silver", "gold", "navy",
    "gray", "brown", "beige", "khaki", "cream", "maroon", "olive"
}
MATERIAL_WORDS = {
    "leather", "steel", "ceramic", "cotton", "wool", "plastic", "glass", "wood",
    "aluminum", "denim", "rubber", "silicone", "canvas", "mesh"
}
CATEGORY_WORDS = {
    "shoe", "shoes", "bag", "backpack", "mug", "kettle", "lamp", "chair",
    "headphones", "watch", "sneaker", "sneakers", "jacket", "phone", "camera",
    "speaker", "mouse", "keyboard", "desk", "bottle", "wallet"
}
ATTRIBUTE_WORDS = {
    "waterproof", "wireless", "compact", "portable", "lightweight", "durable",
    "ergonomic", "foldable", "insulated", "cushioned", "premium", "sturdy",
    "sleek", "minimal", "eco", "soft", "scratch", "resistant"
}
SIZE_WORDS = {"small", "medium", "large", "xl", "xxl", "compact", "mini", "full", "extra"}
BRAND_WORDS = {"nike", "adidas", "sony", "samsung", "apple", "canon", "logitech", "tesla", "gucci", "h&m"}

SLOT_LABEL_MAP = {
    "PAD": 0,
    "BRAND": 1,
    "CATEGORY": 2,
    "COLOR": 3,
    "MATERIAL": 4,
    "ATTRIBUTE": 5,
    "SIZE": 6,
}

PRODUCT_TEMPLATES = [
    "find the {color} {material} {category} for {attribute} use",
    "show me a {color} {category} with {attribute} design",
    "look for the {brand} {category} in {color}",
    "find this {category} in {color} with {material} finish",
    "search for a {size} {color} {category} made of {material}",
    "i need a {attribute} {category} in {color} and {material}",
    "find the {brand} {category} with {attribute} support",
    "show me the {color} {category} for {attribute} everyday use",
    "find this {category} in {color} with {size} profile",
    "search for a {material} {category} that looks {attribute}",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_query(index: int) -> Tuple[str, List[int]]:
    brand = random.choice(list(BRAND_WORDS))
    color = random.choice(list(COLOR_WORDS))
    material = random.choice(list(MATERIAL_WORDS))
    category = random.choice(list(CATEGORY_WORDS))
    attribute = random.choice(list(ATTRIBUTE_WORDS))
    size = random.choice(list(SIZE_WORDS))

    template = random.choice(PRODUCT_TEMPLATES)
    query = template.format(
        brand=brand,
        color=color,
        material=material,
        category=category,
        attribute=attribute,
        size=size,
    )

    tokens = query.lower().split()
    slot_ids: List[int] = []
    for token in tokens:
        if token in BRAND_WORDS:
            slot_ids.append(SLOT_LABEL_MAP["BRAND"])
        elif token in CATEGORY_WORDS:
            slot_ids.append(SLOT_LABEL_MAP["CATEGORY"])
        elif token in COLOR_WORDS:
            slot_ids.append(SLOT_LABEL_MAP["COLOR"])
        elif token in MATERIAL_WORDS:
            slot_ids.append(SLOT_LABEL_MAP["MATERIAL"])
        elif token in ATTRIBUTE_WORDS:
            slot_ids.append(SLOT_LABEL_MAP["ATTRIBUTE"])
        elif token in SIZE_WORDS:
            slot_ids.append(SLOT_LABEL_MAP["SIZE"])
        else:
            slot_ids.append(SLOT_LABEL_MAP["PAD"])

    if len(slot_ids) < 32:
        slot_ids += [SLOT_LABEL_MAP["PAD"]] * (32 - len(slot_ids))
    else:
        slot_ids = slot_ids[:32]

    return query, slot_ids


def generate_image(path: Path, index: int) -> None:
    torch.manual_seed(1000 + index)
    image = Image.new("RGB", (224, 224), color=(20, 24, 32))
    draw = ImageDraw.Draw(image)

    # Background gradient-like effect using torch-generated values.
    for y in range(224):
        for x in range(224):
            r = int(30 + 20 * torch.rand(1).item() + 25 * (x / 223))
            g = int(40 + 20 * torch.rand(1).item() + 18 * (y / 223))
            b = int(70 + 20 * torch.rand(1).item() + 12 * ((x + y) / 446))
            image.putpixel((x, y), (r % 255, g % 255, b % 255))

    # Add product-like shapes and texture.
    shape_color = (
        int(80 + 100 * torch.rand(1).item()),
        int(90 + 90 * torch.rand(1).item()),
        int(110 + 90 * torch.rand(1).item()),
    )
    draw.rectangle((40, 40, 184, 184), outline=shape_color, width=8)
    draw.ellipse((72, 68, 152, 148), fill=shape_color)

    # Add a few smaller accent elements to mimic real catalog images.
    for _ in range(5):
        x0 = int(torch.randint(20, 204, (1,)).item())
        y0 = int(torch.randint(20, 204, (1,)).item())
        x1 = x0 + int(torch.randint(12, 48, (1,)).item())
        y1 = y0 + int(torch.randint(12, 48, (1,)).item())
        draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255, 255))

    image.save(path)


def build_dataset(num_rows: int = 100) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for index in range(num_rows):
        query, slot_ids = build_query(index)
        intent_id = random.randint(0, 4)
        image_name = f"product_{index:03d}.png"
        image_path = IMAGE_DIR / image_name
        generate_image(image_path, index)

        records.append(
            {
                "text_query": query,
                "image_path": str(image_path.relative_to(ROOT)),
                "intent_id": intent_id,
                "slot_ids": slot_ids,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    set_seed(42)
    dataset = build_dataset(100)
    dataset.to_csv(CSV_PATH, index=False)
    print(f"Created {len(dataset)} mock rows and saved metadata to {CSV_PATH}")


if __name__ == "__main__":
    main()
