"""
AI-Generated Object Quality Assessment using CLIP - Contrastive Analysis

This script evaluates AI-generated objects using contrastive comparison:
- Compares each object against "good example" vs "AI failure pattern"
- Uses specific failure modes that AI commonly produces:
  * Fork: Wrong number of prongs, merged tines
  * Knife: Blended blade-handle, wrong proportions
  * Bowl: Jagged edges, asymmetrical shape
  * Glass: Opacity issues, impossible geometry
  * etc.

Quality score represents probability of resembling good example vs AI failure.
Higher scores (closer to 1.0) indicate more realistic objects.
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

# Define AI-specific failure patterns for contrastive evaluation
AI_QUALITY_PROMPTS = {
    "fork": {
        "good_example": "a well-formed fork with exactly 3 or 4 distinct prongs and a clear handle",
        "ai_failure": "a malformed fork with wrong number of prongs, merged tines, or distorted shape"
    },
    "bowl": {
        "good_example": "a properly shaped bowl with smooth curved rim and consistent circular outline",
        "ai_failure": "a deformed bowl with jagged edges, asymmetrical shape, or broken rim"
    },
    "napkin": {
        "good_example": "a realistic napkin with proper fabric folds and rectangular shape",
        "ai_failure": "an artificial napkin with impossible geometry, strange texture, or unnatural folds"
    },
    "glass": {
        "good_example": "a transparent glass with smooth walls, clear rim, and realistic reflections",
        "ai_failure": "an opaque or distorted glass with impossible shape or unrealistic appearance"
    },
    "knive": {
        "good_example": "a realistic knife with distinct sharp blade and separate handle",
        "ai_failure": "a malformed knife with blended blade-handle, wrong proportions, or impossible geometry"
    },
    "plate": {
        "good_example": "a round plate with smooth circular rim and flat even surface",
        "ai_failure": "a deformed plate with uneven edges, warped surface, or non-circular shape"
    },
    "plate_dirt": {
        "good_example": "a round plate with realistic food stains and proper circular rim",
        "ai_failure": "a plate with artificial-looking residues, impossible stains, or deformed shape"
    },
    "bowl_dirt": {
        "good_example": "a bowl with realistic food residues inside and proper curved rim",
        "ai_failure": "a bowl with fake-looking residues, impossible stains, or distorted shape"
    }
}

def get_ai_quality_score(image_tensor, true_label, model):
    """
    Evaluate AI-generated object quality using contrastive analysis.
    Compares against specific AI failure patterns.
    Returns a score between 0 and 1, where 1 means realistic/well-formed.
    """
    if true_label not in AI_QUALITY_PROMPTS:
        return 0.5, 0.0, 0.0, "unknown_class"
    
    # Get contrastive prompts for this object type
    prompts = AI_QUALITY_PROMPTS[true_label]
    
    # Create specific prompts
    good_prompt = prompts["good_example"]
    failure_prompt = prompts["ai_failure"]
    
    text_inputs = clip.tokenize([good_prompt, failure_prompt]).to(device)
    
    with torch.no_grad():
        # Get image features
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Get text features
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (image_features @ text_features.T).squeeze(0)
        good_sim = similarities[0].item()
        failure_sim = similarities[1].item()
        
        # Use softmax for probability
        exp_good = torch.exp(similarities[0])
        exp_failure = torch.exp(similarities[1])
        quality_probability = (exp_good / (exp_good + exp_failure)).item()
        
        return quality_probability, good_sim, failure_sim

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

# Define quality thresholds - more stringent for contrastive scoring
HIGH_QUALITY_THRESHOLD = 0.65   # Clearly resembles good example
MEDIUM_QUALITY_THRESHOLD = 0.35  # Somewhat ambiguous

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

        # Get AI quality score using contrastive analysis
        quality_score, good_sim, failure_sim = get_ai_quality_score(crop_input, true_label, model)
        
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
            "good_similarity": round(good_sim, 4),
            "failure_similarity": round(failure_sim, 4),
            "confidence": round(abs(good_sim - failure_sim), 4)  # How confident the assessment is
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
            "good_similarity": round(good_sim, 4),
            "failure_similarity": round(failure_sim, 4),
            "confidence": round(abs(good_sim - failure_sim), 4)
        })

        print(result)  # or store/save later

# Additional analysis - save images by confidence level for manual inspection
print(f"\n=== Analysis Summary ===")
total_objects = len(results)
high_quality = sum(1 for r in results if r["quality_category"] == "high")
medium_quality = sum(1 for r in results if r["quality_category"] == "medium") 
low_quality = sum(1 for r in results if r["quality_category"] == "low")

print(f"Total objects analyzed: {total_objects}")
print(f"High quality: {high_quality} ({high_quality/total_objects*100:.1f}%)")
print(f"Medium quality: {medium_quality} ({medium_quality/total_objects*100:.1f}%)")
print(f"Low quality: {low_quality} ({low_quality/total_objects*100:.1f}%)")

# Find most/least confident assessments
if results:
    most_confident = max(results, key=lambda x: x["confidence"])
    least_confident = min(results, key=lambda x: x["confidence"])
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    
    print(f"\nAverage confidence: {avg_confidence:.3f}")
    print(f"Most confident: {most_confident['true_label']} (confidence: {most_confident['confidence']}, quality: {most_confident['quality_category']})")
    print(f"Least confident: {least_confident['true_label']} (confidence: {least_confident['confidence']}, quality: {least_confident['quality_category']})")

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
    "methodology": "Contrastive AI quality assessment - good examples vs failure patterns",
    "per_image_results": image_summary,
    "detailed_results": results
}

with open("clip_ai_feature_output.json", "w") as f:
    json.dump(output_data, f, indent=2)

with open("clip_ai_feature_summary.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image", "Total", "High_Quality", "Medium_Quality", "Low_Quality", "Avg_Quality", "High_Percent", "Avg_Confidence"])

    for img_path, stats in image_summary.items():
        high_percent = (stats["high_quality_count"] / stats["total"] * 100) if stats["total"] > 0 else 0
        avg_confidence = sum(item["confidence"] for item in stats["items"]) / len(stats["items"]) if stats["items"] else 0
        writer.writerow([
            img_path, 
            stats["total"], 
            stats["high_quality_count"], 
            stats["medium_quality_count"], 
            stats["low_quality_count"],
            f"{stats['avg_quality']:.3f}",
            f"{high_percent:.1f}%",
            f"{avg_confidence:.3f}"
        ])
