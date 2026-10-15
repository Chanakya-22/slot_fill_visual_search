"""Procedural product catalogue and query generation for synthetic experiments."""

import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.labels import INTENT_NAME_TO_ID, SLOT_TAG_TO_ID

DEFAULT_NUM_PRODUCTS = 100
RENDERER_NAME = "procedural_pillow_vector_v1"

CATEGORIES = {
    "gear": {
        "phrases": ("gear", "gear assembly"),
        "shapes": ("radial tooth", "spiral tooth", "beveled tooth", "fine tooth"),
    },
    "bracket": {
        "phrases": ("bracket", "mounting bracket"),
        "shapes": ("L shaped", "U shaped", "angled plate", "offset plate"),
    },
    "hinge": {
        "phrases": ("hinge", "pivot hinge"),
        "shapes": ("butt style", "strap style", "offset leaf", "compact leaf"),
    },
    "connector": {
        "phrases": ("connector", "electrical connector"),
        "shapes": ("round body", "rectangular body", "ribbed body", "flanged body"),
    },
    "mount": {
        "phrases": ("mount", "equipment mount"),
        "shapes": ("flat plate", "raised arm", "corner plate", "extended arm"),
    },
}

COLORS = {
    "red": (190, 45, 43),
    "navy blue": (39, 70, 126),
    "forest green": (48, 112, 76),
    "orange": (222, 112, 39),
    "silver": (153, 165, 172),
    "black": (58, 62, 66),
    "teal": (37, 137, 137),
    "yellow": (220, 176, 48),
    "matte black": (48, 51, 54),
    "light gray": (170, 177, 181),
}
MATERIALS = ("stainless steel", "aluminum", "brass", "nylon", "titanium")
COMPATIBLE_CATEGORIES = {
    "gear": "mount",
    "bracket": "hinge",
    "hinge": "bracket",
    "connector": "mount",
    "mount": "connector",
}

QUERY_TEMPLATES = (
    ("search_product", "find a {color} {shape} {category}"),
    ("search_product", "show me the {category} with a {shape} profile in {color}"),
    ("search_product", "search for a {color} {category} with a {shape} design"),
    ("search_product", "looking for the {shape} style {category} in {color}"),
    ("troubleshoot_product", "how can I inspect the {color} {shape} {category} for wear"),
    ("troubleshoot_product", "what should I check if this {shape} {category} in {color} seems loose"),
    ("troubleshoot_product", "how do I troubleshoot a {color} {category} with a {shape} form"),
    ("troubleshoot_product", "what could cause this {shape} {category} in {color} to misalign"),
    ("identify_product", "identify the {color} {shape} {category} shown here"),
    ("identify_product", "what kind of {category} has this {shape} shape and {color} finish"),
    ("identify_product", "tell me whether this is a {color} {shape} {category}"),
    ("identify_product", "which {category} is pictured with a {color} finish and {shape} body"),
    ("check_compatibility", "will this {color} {shape} {category} fit a {compatible_category}"),
    ("check_compatibility", "check if the {shape} {category} in {color} works with a {compatible_category}"),
    ("check_compatibility", "can I pair this {color} {category} with a {compatible_category} if it has a {shape} form"),
    ("check_compatibility", "does the {compatible_category} fit the {shape} {category} with a {color} finish"),
    ("request_specifications", "show specs for the {color} {shape} {category} made of {material}"),
    ("request_specifications", "what are the specifications of this {material} {category} with a {shape} profile in {color}"),
    ("request_specifications", "list dimensions and material for the {color} {shape} {category} made of {material}"),
    ("request_specifications", "give me the material and dimensions for this {shape} {category} in {color} made of {material}"),
    ("request_specifications", "what is the price of the {color} {shape} {category} listed at {price}"),
    ("request_specifications", "confirm whether the {shape} {category} in {color} costs {price}"),
    ("request_specifications", "check the listed price {price} for this {color} {shape} {category}"),
    ("request_specifications", "how much is the {color} {shape} {category} priced at {price}"),
)
_PLACEHOLDER = re.compile(r"{([a-z_]+)}")
_TEMPLATE_IDS_BY_INTENT = {
    intent: tuple(index for index, (template_intent, _) in enumerate(QUERY_TEMPLATES) if template_intent == intent)
    for intent in INTENT_NAME_TO_ID
}


def _tag_type(field: str) -> str:
    if field in ("category", "compatible_category"):
        return "CATEGORY"
    return field.upper()


def _render_query(template: str, fields: dict[str, str]) -> tuple[list[str], list[str]]:
    tokens: list[str] = []
    tags: list[str] = []
    cursor = 0

    for match in _PLACEHOLDER.finditer(template):
        literal_tokens = template[cursor : match.start()].split()
        tokens.extend(literal_tokens)
        tags.extend(["O"] * len(literal_tokens))

        field = match.group(1)
        value_tokens = fields[field].split()
        entity_type = _tag_type(field)
        tokens.extend(value_tokens)
        tags.extend([f"B-{entity_type}"] + [f"I-{entity_type}"] * (len(value_tokens) - 1))
        cursor = match.end()

    trailing_tokens = template[cursor:].split()
    tokens.extend(trailing_tokens)
    tags.extend(["O"] * len(trailing_tokens))
    return tokens, tags


def render_product_image(product: dict[str, Any], image_path: Path) -> None:
    """Draw a category-specific hardware silhouette with its recorded finish colour."""
    image_size = 224
    image = Image.new("RGB", (image_size, image_size), (239, 241, 242))
    draw = ImageDraw.Draw(image)
    color = COLORS[product["color"]]
    dark = tuple(max(channel - 55, 0) for channel in color)
    light = tuple(min(channel + 45, 255) for channel in color)
    center_x = 112
    center_y = 111
    variation = product["shape_index"]

    draw.ellipse((48, 166, 176, 190), fill=(211, 214, 215))

    category = product["category"]
    if category == "gear":
        points = []
        tooth_count = 12 + variation * 2
        for point_index in range(tooth_count * 2):
            angle = point_index * math.pi / tooth_count
            radius = 66 if point_index % 2 == 0 else 53
            points.append((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
        draw.polygon(points, fill=color, outline=dark)
        draw.ellipse((79, 78, 145, 144), fill=(239, 241, 242), outline=dark, width=5)
        draw.ellipse((99, 98, 125, 124), fill=(199, 203, 205), outline=dark, width=3)
        for spoke in range(variation + 2):
            draw.line((112, 53, 112, 78), fill=light, width=4)
            if spoke:
                draw.line((112, 78, 112 + spoke * 7, 82), fill=light, width=3)
    elif category == "bracket":
        if variation == 0:
            draw.line((66, 61, 66, 157, 157, 157), fill=dark, width=31, joint="curve")
            draw.line((70, 63, 70, 151, 153, 151), fill=color, width=23, joint="curve")
        elif variation == 1:
            draw.line((61, 68, 61, 151, 163, 151, 163, 68), fill=dark, width=27, joint="curve")
            draw.line((64, 68, 64, 147, 160, 147, 160, 68), fill=color, width=19, joint="curve")
        elif variation == 2:
            draw.polygon(((58, 157), (58, 68), (80, 68), (155, 141), (155, 157)), fill=color, outline=dark)
            draw.line((80, 81, 137, 137), fill=light, width=5)
        else:
            draw.line((57, 69, 57, 142, 106, 142, 106, 157, 169, 157), fill=dark, width=29, joint="curve")
            draw.line((61, 70, 61, 137, 111, 137, 111, 151, 165, 151), fill=color, width=20, joint="curve")
        hole_positions = (
            ((66, 84), (66, 127), (105, 151), (146, 151))
            if variation == 0
            else ((61, 89), (61, 128), (163, 89), (163, 128))
            if variation == 1
            else ((66, 83), (66, 123), (142, 149))
            if variation == 2
            else ((58, 88), (58, 121), (143, 151), (163, 151))
        )
        for x, y in hole_positions:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(239, 241, 242), outline=dark, width=2)
    elif category == "hinge":
        offset = variation * 4
        draw.rounded_rectangle((52, 63, 105 + offset, 160), radius=8, fill=color, outline=dark, width=4)
        draw.rounded_rectangle((119 - offset, 63, 172, 160), radius=8, fill=color, outline=dark, width=4)
        draw.ellipse((103, 53, 121, 171), fill=(181, 187, 190), outline=dark, width=3)
        for x in (73, 146):
            for y in (86, 137):
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(239, 241, 242), outline=dark, width=2)
    elif category == "connector":
        if variation == 0:
            draw.rounded_rectangle((66, 70, 158, 151), radius=24, fill=color, outline=dark, width=5)
        elif variation == 1:
            draw.rectangle((66, 70, 158, 151), fill=color, outline=dark, width=5)
        elif variation == 2:
            draw.rounded_rectangle((66, 70, 158, 151), radius=16, fill=color, outline=dark, width=5)
            for rib_x in (85, 105, 125, 145):
                draw.line((rib_x, 77, rib_x, 144), fill=light, width=4)
        else:
            draw.polygon(((77, 66), (148, 66), (163, 82), (163, 139), (148, 155), (77, 155), (61, 139), (61, 82)), fill=color, outline=dark)
        for pin in range(3 + variation):
            x = 83 + pin * (58 // (2 + variation))
            draw.rectangle((x, 145, x + 7, 174), fill=(191, 195, 197), outline=dark)
        draw.line((78, 86, 145, 86), fill=light, width=5)
    else:
        draw.rounded_rectangle((52, 92, 172, 151), radius=10, fill=color, outline=dark, width=5)
        if variation == 0:
            pass
        elif variation == 1:
            draw.polygon(((92, 95), (105, 56), (126, 56), (140, 95)), fill=color, outline=dark)
        elif variation == 2:
            draw.polygon(((52, 92), (92, 92), (92, 54), (112, 54), (112, 112), (172, 112), (172, 133), (72, 133)), fill=color, outline=dark)
        else:
            draw.polygon(((101, 95), (101, 47), (123, 47), (137, 95)), fill=color, outline=dark)
        for x in (72, 151):
            draw.ellipse((x - 8, 108, x + 8, 124), fill=(239, 241, 242), outline=dark, width=3)

    draw.line((76, 71, 96, 64), fill=light, width=4)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path, format="PNG", optimize=False)


def _build_catalogue(num_products: int, rng: random.Random) -> list[dict[str, Any]]:
    category_names = tuple(CATEGORIES)
    if num_products < len(category_names) or num_products % len(category_names) != 0:
        raise ValueError(f"num_products must be a positive multiple of {len(category_names)}.")

    products = []
    products_per_category = num_products // len(category_names)
    for category_index, (category, spec) in enumerate(CATEGORIES.items()):
        combinations = [(color, shape_index) for color in COLORS for shape_index in range(len(spec["shapes"]))]
        rng.shuffle(combinations)
        for local_index, (color, shape_index) in enumerate(combinations[:products_per_category]):
            product_number = category_index * products_per_category + local_index
            products.append(
                {
                    "product_id": f"product_{product_number:04d}",
                    "category": category,
                    "category_phrase": spec["phrases"][product_number % len(spec["phrases"])],
                    "shape": spec["shapes"][shape_index],
                    "shape_index": shape_index,
                    "color": color,
                    "material": rng.choice(MATERIALS),
                    "price_cents": rng.randrange(650, 25001, 25),
                    "compatible_with": [COMPATIBLE_CATEGORIES[category]],
                }
            )
    return products


def _example_records(
    products: list[dict[str, Any]], image_dir: Path, seed: int
) -> list[dict[str, Any]]:
    records = []
    for product_number, product in enumerate(products):
        product["image_path"] = f"images/{product['product_id']}.png"
        render_product_image(product, image_dir / f"{product['product_id']}.png")
        fields = {
            "category": product["category_phrase"],
            "compatible_category": CATEGORIES[product["compatible_with"][0]]["phrases"][0],
            "color": product["color"],
            "shape": product["shape"],
            "material": product["material"],
            "price": f"${product['price_cents'] / 100:.2f}",
        }
        for intent in INTENT_NAME_TO_ID:
            template_indices = _TEMPLATE_IDS_BY_INTENT[intent]
            chosen_indices = (
                template_indices[product_number % len(template_indices)],
                template_indices[(product_number + 1) % len(template_indices)],
            )
            for template_index in chosen_indices:
                _, template = QUERY_TEMPLATES[template_index]
                tokens, slot_tags = _render_query(template, fields)
                records.append(
                    {
                        "sample_id": f"sample_{len(records):05d}",
                        "product_id": product["product_id"],
                        "image_path": product["image_path"],
                        "text_query": " ".join(tokens),
                        "intent": intent,
                        "intent_id": INTENT_NAME_TO_ID[intent],
                        "tokens": json.dumps(tokens, ensure_ascii=True),
                        "slot_tags": json.dumps(slot_tags, ensure_ascii=True),
                        "slot_ids": json.dumps([SLOT_TAG_TO_ID[tag] for tag in slot_tags]),
                        "product_metadata": json.dumps(product, sort_keys=True, ensure_ascii=True),
                        "generation_info": json.dumps(
                            {
                                "seed": seed,
                                "renderer": RENDERER_NAME,
                                "template_id": f"template_{template_index:02d}",
                            },
                            sort_keys=True,
                        ),
                    }
                )
    return records


def generate_synthetic_dataset(
    output_dir: Path | str,
    seed: int = 42,
    num_products: int = DEFAULT_NUM_PRODUCTS,
) -> dict[str, Any]:
    """Write a deterministic catalogue, procedural images, query CSV, and summary."""
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    products = _build_catalogue(num_products, rng)
    records = _example_records(products, image_dir, seed)

    csv_path = output_dir / "dataset.csv"
    fieldnames = (
        "image_path",
        "text_query",
        "intent_id",
        "slot_ids",
        "sample_id",
        "product_id",
        "intent",
        "tokens",
        "slot_tags",
        "product_metadata",
        "generation_info",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as dataset_file:
        writer = csv.DictWriter(dataset_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    catalogue_path = output_dir / "catalogue.json"
    catalogue_path.write_text(
        json.dumps(products, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    intent_counts = Counter(record["intent"] for record in records)
    slot_counts = Counter(
        tag
        for record in records
        for tag in json.loads(record["slot_tags"])
    )
    summary = {
        "seed": seed,
        "num_examples": len(records),
        "num_unique_products": len(products),
        "num_query_templates": len(QUERY_TEMPLATES),
        "intent_distribution": dict(sorted(intent_counts.items())),
        "slot_label_distribution": dict(sorted(slot_counts.items())),
        "image_generation_method": RENDERER_NAME,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary
