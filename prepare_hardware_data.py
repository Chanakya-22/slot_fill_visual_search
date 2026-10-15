import json
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

def generate_mock_data(num_samples=500):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "data", "raw")
    images_dir = os.path.join(output_dir, "images")
    
    os.makedirs(images_dir, exist_ok=True)

    MATERIALS = ["steel", "titanium", "aluminum", "plastic", "copper"]
    ITEMS = ["bracket", "gear", "mount", "hinge", "connector"]
    
    data = []
    for i in range(num_samples):
        img_filename = f"sample_{i:03d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        
        # Generate synthetic image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(img_path)

        item = random.choice(ITEMS)
        material = random.choice(MATERIALS)
        
        # Query format: "find compatible [item] in [material]"
        query_words = ["find", "compatible", item, "in", material]
        query = " ".join(query_words)
        intent_id = i % 5  # Distribute across 5 intents
        
        # 0: O, 1: B-Item, 2: B-Material
        slot_ids = [0, 0, 1, 0, 2]

        data.append({
            "image_path": img_path,
            "text_query": query,
            "intent_id": intent_id,
            "slot_ids": json.dumps(slot_ids)
        })

    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated {num_samples} structured hardware samples at '{csv_path}'.")

if __name__ == "__main__":
    generate_mock_data()