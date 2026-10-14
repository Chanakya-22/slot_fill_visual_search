import json
import os
import numpy as np
import pandas as pd
from PIL import Image

def generate_mock_data(num_samples=100):
    # Resolve absolute path relative to this script file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "data", "raw")
    images_dir = os.path.join(output_dir, "images")
    
    os.makedirs(images_dir, exist_ok=True)

    queries = [
        "find this gear in titanium",
        "search for compatible mount in steel",
        "get replacement bracket aluminum",
        "find similar connector heavy duty",
        "locate matching hinge stainless steel"
    ]

    data = []
    for i in range(num_samples):
        img_filename = f"sample_{i:03d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        
        # Generate synthetic image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(img_path)

        query = queries[i % len(queries)]
        intent_id = i % 5
        num_words = len(query.split())
        slot_ids = list(np.random.randint(0, 4, size=num_words))

        data.append({
            "image_path": img_path,  # Stores absolute path
            "text_query": query,
            "intent_id": intent_id,
            "slot_ids": json.dumps([int(x) for x in slot_ids])
        })

    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated {num_samples} mock samples at '{csv_path}'.")

if __name__ == "__main__":
    generate_mock_data()