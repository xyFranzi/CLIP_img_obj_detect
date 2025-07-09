import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
import datetime
import csv
from collections import defaultdict

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'data')
images_dir = os.path.join(data_dir, 'img')
bboxes_dir = os.path.join(data_dir, 'box')

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Define realistic description templates for each class
REALISTIC_DESCRIPTIONS = {
    "fork": "a clear, well-visible fork",
    "bowl": "a clear, well-visible bowl", 
    "napkin": "a clear, well-visible napkin",
    "glass": "a clear, well-visible glass",
    "knive": "a clear, well-visible knife",
    "plate": "a clear, well-visible plate",
    "plate_dirt": "a clear, well-visible plate with food residues",
    "bowl_dirt": "a clear, well-visible bowl with food residues"
}

# Define contrasting descriptions for comparison
UNREALISTIC_DESCRIPTIONS = {
    "fork": "an unclear, blurry, or unrecognizable fork",
    "bowl": "an unclear, blurry, or unrecognizable bowl",
    "napkin": "an unclear, blurry, or unrecognizable napkin", 
    "glass": "an unclear, blurry, or unrecognizable glass",
    "knive": "an unclear, blurry, or unrecognizable knife",
    "plate": "an unclear, blurry, or unrecognizable plate",
    "plate_dirt": "an unclear, blurry, or unrecognizable plate with food residues",
    "bowl_dirt": "an unclear, blurry, or unrecognizable bowl with food residues"
}

def get_realism_score(image_tensor, true_label, model):
    """
    Calculate how realistic/clear the cropped image is for the given true label.
    Returns a score between 0 and 1, where 1 means very realistic/clear.
    """
    if true_label not in REALISTIC_DESCRIPTIONS:
        return 0.0, "unknown_class"
    
    # Create text prompts for realistic vs unrealistic
    realistic_prompt = REALISTIC_DESCRIPTIONS[true_label]
    unrealistic_prompt = UNREALISTIC_DESCRIPTIONS[true_label]
    
    text_inputs = clip.tokenize([realistic_prompt, unrealistic_prompt]).to(device)
    
    with torch.no_grad():
        # Get image features
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Get text features
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (image_features @ text_features.T).squeeze(0)
        realistic_score = similarities[0].item()
        unrealistic_score = similarities[1].item()
        
        # Convert to 0-1 probability using softmax-like approach
        exp_realistic = torch.exp(similarities[0])
        exp_unrealistic = torch.exp(similarities[1])
        realism_probability = (exp_realistic / (exp_realistic + exp_unrealistic)).item()
        
        return realism_probability, realistic_score, unrealistic_score

# Padding function
def apply_padding(x1, y1, x2, y2, padding, img_width, img_height):
    x1_p = max(0, x1 - padding)
    y1_p = max(0, y1 - padding)
    x2_p = min(img_width, x2 + padding)
    y2_p = min(img_height, y2 + padding)
    return x1_p, y1_p, x2_p, y2_p

# Inference loop
results = []

# === Build image-to-bbox mapping (required for imagedirmap) ===
image_files = sorted(f for f in os.listdir(images_dir) if f.endswith('.png'))
imagedirmap = {}

for image_file in image_files:
    try:
        image_num = int(image_file.split('__')[1].split('_')[0])
    except (IndexError, ValueError):
        print(f"Skipping unrecognized image file format: {image_file}")
        continue

    bbox_num = image_num - 1
    bbox_file = f"bounding_boxes_{bbox_num:05d}.json"

    image_path = os.path.join('data/img', image_file)
    bbox_path = os.path.join('data/box', bbox_file)

    imagedirmap[image_path] = bbox_path

image_summary = defaultdict(lambda: {
    "high_realism_count": 0,  # realism_score >= 0.7
    "medium_realism_count": 0,  # 0.4 <= realism_score < 0.7
    "low_realism_count": 0,  # realism_score < 0.4
    "total": 0,
    "avg_realism": 0.0,
    "items": []
})

# Define realism thresholds
HIGH_REALISM_THRESHOLD = 0.7
MEDIUM_REALISM_THRESHOLD = 0.4

for img_path, bbox_path in tqdm(imagedirmap.items()):
    full_img_path = os.path.join(current_dir, img_path)
    full_bbox_path = os.path.join(current_dir, bbox_path)

    image = Image.open(full_img_path).convert("RGB")
    width, height = image.size

    with open(full_bbox_path, 'r') as f:
        boxes = json.load(f)

    for box in boxes:
        true_label = box["name"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        # Apply padding
        pad = 15  # pixels
        x1_p, y1_p, x2_p, y2_p = apply_padding(x1, y1, x2, y2, pad, width, height)

        # Crop and preprocess
        crop = image.crop((x1_p, y1_p, x2_p, y2_p))
        crop_input = preprocess(crop).unsqueeze(0).to(device)

        # Get realism score based on true label
        realism_score, realistic_sim, unrealistic_sim = get_realism_score(crop_input, true_label, model)
        
        # Categorize realism level
        if realism_score >= HIGH_REALISM_THRESHOLD:
            realism_category = "high"
        elif realism_score >= MEDIUM_REALISM_THRESHOLD:
            realism_category = "medium"
        else:
            realism_category = "low"

        # save_dir = "cropped_regions"
        # os.makedirs(save_dir, exist_ok=True)
        # crop.save(os.path.join(save_dir, f"{true_label}_{realism_category}_{x1}_{y1}.png"))

        result = {
            "image": img_path,
            "true_label": true_label,
            "realism_score": round(realism_score, 4),
            "realism_category": realism_category,
            "realistic_similarity": round(realistic_sim, 4),
            "unrealistic_similarity": round(unrealistic_sim, 4),
        }
        results.append(result)

        # Update per-image summary
        image_summary[img_path]["total"] += 1
        if realism_category == "high":
            image_summary[img_path]["high_realism_count"] += 1
        elif realism_category == "medium":
            image_summary[img_path]["medium_realism_count"] += 1
        else:
            image_summary[img_path]["low_realism_count"] += 1
            
        image_summary[img_path]["items"].append({
            "true_label": true_label,
            "realism_score": round(realism_score, 4),
            "realism_category": realism_category,
            "realistic_similarity": round(realistic_sim, 4),
            "unrealistic_similarity": round(unrealistic_sim, 4)
        })

        print(result)  # or store/save later

# Calculate average realism scores for each image
for img_path, stats in image_summary.items():
    if stats["total"] > 0:
        total_realism = sum(item["realism_score"] for item in stats["items"])
        stats["avg_realism"] = total_realism / stats["total"]

output_data = {
    "timestamp": datetime.datetime.now().isoformat(),
    "realism_thresholds": {
        "high": HIGH_REALISM_THRESHOLD,
        "medium": MEDIUM_REALISM_THRESHOLD
    },
    "per_image_results": image_summary,
    "detailed_results": results
}

with open("clip_verification_input_both_output.json", "w") as f:
    json.dump(output_data, f, indent=2)

with open("clip_verification_input_both_summary.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image", "Total", "High_Realism", "Medium_Realism", "Low_Realism", "Avg_Realism", "High_Percent"])

    for img_path, stats in image_summary.items():
        high_percent = (stats["high_realism_count"] / stats["total"] * 100) if stats["total"] > 0 else 0
        writer.writerow([
            img_path, 
            stats["total"], 
            stats["high_realism_count"], 
            stats["medium_realism_count"], 
            stats["low_realism_count"],
            f"{stats['avg_realism']:.3f}",
            f"{high_percent:.1f}%"
        ])
