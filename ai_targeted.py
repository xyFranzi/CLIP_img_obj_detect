"""
AI-Generated Object Quality Assessment using CLIP

This script evaluates the structural quality and accuracy of AI-generated objects
by checking for specific anatomical features:

- Fork: Proper number of tines (3-4), clear handle separation
- Knife: Distinct blade and handle, proper proportions  
- Bowl: Curved rim, proper shape, visible interior
- Glass: Transparency, cylindrical shape, clear rim
- Plate: Circular rim, flat surface, proper proportions
- Napkin: Fabric texture, rectangular/folded shape

The quality score is calculated using a 3-way comparison:
- High quality: Complete structure with all expected features
- Medium quality: Recognizable but some distortion/unclear parts
- Low quality: Incomplete, deformed, or unrecognizable structure
"""

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

# Define quality criteria for AI-generated objects - focusing on structural accuracy
QUALITY_DESCRIPTIONS = {
    "fork": {
        "high_quality": "a complete fork with 3-4 visible tines and a clear handle",
        "medium_quality": "a fork with visible tines but some parts unclear or partially hidden", 
        "low_quality": "an incomplete fork with missing tines, deformed shape, or unrecognizable structure"
    },
    "bowl": {
        "high_quality": "a complete bowl with clear curved rim, proper circular or oval shape, and visible interior",
        "medium_quality": "a bowl with recognizable shape but some distortion or partially obscured edges",
        "low_quality": "an incomplete bowl with broken rim, impossible geometry, or unrecognizable as a bowl"
    },
    "napkin": {
        "high_quality": "a complete napkin with clear fabric texture, proper rectangular or folded shape",
        "medium_quality": "a napkin with recognizable shape but unclear texture or slightly distorted form",
        "low_quality": "an incomplete napkin with impossible folds, unclear material, or unrecognizable shape"
    },
    "glass": {
        "high_quality": "a complete glass with clear transparent appearance, proper cylindrical shape, and visible rim",
        "medium_quality": "a glass with recognizable shape but some opacity issues or slight distortion",
        "low_quality": "an incomplete glass with impossible geometry, no transparency, or unrecognizable structure"
    },
    "knive": {
        "high_quality": "a complete knife with clearly separated blade and handle, proper proportions",
        "medium_quality": "a knife with visible blade and handle but some unclear separation or proportions",
        "low_quality": "an incomplete knife with merged blade-handle, impossible shape, or unrecognizable structure"
    },
    "plate": {
        "high_quality": "a complete plate with clear circular rim, flat surface, and proper proportions",
        "medium_quality": "a plate with recognizable circular shape but some edge distortion or unclear surface",
        "low_quality": "an incomplete plate with broken rim, impossible geometry, or unrecognizable as a plate"
    },
    "plate_dirt": {
        "high_quality": "a complete plate with clear circular rim and realistic food residues or stains",
        "medium_quality": "a plate with recognizable shape and some visible residues but unclear details",
        "low_quality": "an incomplete plate with unrealistic residues, impossible geometry, or unrecognizable structure"
    },
    "bowl_dirt": {
        "high_quality": "a complete bowl with clear curved rim and realistic food residues inside",
        "medium_quality": "a bowl with recognizable shape and some visible residues but unclear details", 
        "low_quality": "an incomplete bowl with unrealistic residues, broken rim, or unrecognizable structure"
    }
}

def get_quality_score(image_tensor, true_label, model):
    """
    Evaluate the structural quality and accuracy of AI-generated objects.
    Returns quality scores for high/medium/low quality levels.
    """
    if true_label not in QUALITY_DESCRIPTIONS:
        return 0.0, 0.0, 0.0, 0.0, "unknown_class"
    
    # Get quality descriptions for this object type
    quality_prompts = QUALITY_DESCRIPTIONS[true_label]
    
    # Create text prompts for all quality levels
    high_quality_prompt = quality_prompts["high_quality"]
    medium_quality_prompt = quality_prompts["medium_quality"] 
    low_quality_prompt = quality_prompts["low_quality"]
    
    text_inputs = clip.tokenize([
        high_quality_prompt, 
        medium_quality_prompt, 
        low_quality_prompt
    ]).to(device)
    
    with torch.no_grad():
        # Get image features
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Get text features
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (image_features @ text_features.T).squeeze(0)
        high_score = similarities[0].item()
        medium_score = similarities[1].item()
        low_score = similarities[2].item()
        
        # Use softmax to get probabilities
        exp_scores = torch.exp(similarities)
        probabilities = exp_scores / exp_scores.sum()
        
        # Calculate weighted quality score (higher is better)
        # High quality = 1.0, Medium = 0.5, Low = 0.0
        weighted_score = (probabilities[0].item() * 1.0 + 
                         probabilities[1].item() * 0.5 + 
                         probabilities[2].item() * 0.0)
        
        return weighted_score, high_score, medium_score, low_score

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
    "high_quality_count": 0,    # quality_score >= 0.7
    "medium_quality_count": 0,  # 0.4 <= quality_score < 0.7
    "low_quality_count": 0,     # quality_score < 0.4
    "total": 0,
    "avg_quality": 0.0,
    "items": []
})

# Define quality thresholds based on weighted score
HIGH_QUALITY_THRESHOLD = 0.7   # Mostly high-quality features
MEDIUM_QUALITY_THRESHOLD = 0.4  # Mixed or medium features

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

        # Get quality score based on structural accuracy
        quality_score, high_sim, medium_sim, low_sim = get_quality_score(crop_input, true_label, model)
        
        # Categorize quality level
        if quality_score >= HIGH_QUALITY_THRESHOLD:
            quality_category = "high"
        elif quality_score >= MEDIUM_QUALITY_THRESHOLD:
            quality_category = "medium"
        else:
            quality_category = "low"

        # save_dir = "cropped_regions"
        # os.makedirs(save_dir, exist_ok=True)
        # crop.save(os.path.join(save_dir, f"{true_label}_{quality_category}_{x1}_{y1}.png"))

        result = {
            "image": img_path,
            "true_label": true_label,
            "quality_score": round(quality_score, 4),
            "quality_category": quality_category,
            "high_quality_similarity": round(high_sim, 4),
            "medium_quality_similarity": round(medium_sim, 4),
            "low_quality_similarity": round(low_sim, 4),
        }
        results.append(result)

        # Update per-image summary
        image_summary[img_path]["total"] += 1
        if quality_category == "high":
            image_summary[img_path]["high_quality_count"] += 1
        elif quality_category == "medium":
            image_summary[img_path]["medium_quality_count"] += 1
        else:
            image_summary[img_path]["low_quality_count"] += 1
            
        image_summary[img_path]["items"].append({
            "true_label": true_label,
            "quality_score": round(quality_score, 4),
            "quality_category": quality_category,
            "high_quality_similarity": round(high_sim, 4),
            "medium_quality_similarity": round(medium_sim, 4),
            "low_quality_similarity": round(low_sim, 4)
        })

        print(result)  # or store/save later

# Calculate average quality scores for each image
for img_path, stats in image_summary.items():
    if stats["total"] > 0:
        total_quality = sum(item["quality_score"] for item in stats["items"])
        stats["avg_quality"] = total_quality / stats["total"]

output_data = {
    "timestamp": datetime.datetime.now().isoformat(),
    "quality_thresholds": {
        "high": HIGH_QUALITY_THRESHOLD,
        "medium": MEDIUM_QUALITY_THRESHOLD
    },
    "methodology": "Structural quality assessment for AI-generated objects",
    "per_image_results": image_summary,
    "detailed_results": results
}

with open("clip_eval_ai_output.json", "w") as f:
    json.dump(output_data, f, indent=2)

with open("clip_eval_ai_summary.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image", "Total", "High_Quality", "Medium_Quality", "Low_Quality", "Avg_Quality", "High_Percent"])

    for img_path, stats in image_summary.items():
        high_percent = (stats["high_quality_count"] / stats["total"] * 100) if stats["total"] > 0 else 0
        writer.writerow([
            img_path, 
            stats["total"], 
            stats["high_quality_count"], 
            stats["medium_quality_count"], 
            stats["low_quality_count"],
            f"{stats['avg_quality']:.3f}",
            f"{high_percent:.1f}%"
        ])
