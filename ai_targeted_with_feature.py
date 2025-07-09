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
# Define AI-specific failure patterns for contrastive evaluation
AI_QUALITY_PROMPTS = {
    "fork": {
        "good_examples": [
            "a perfectly realistic fork with three sharp distinct prongs",
            "a high-quality photorealistic fork with clear tines",
            "a real metal fork with proper proportions"
        ],
        "ai_failures": [
            "a blurry distorted fork with merged tines",
            "an artificial-looking fork with wrong number of prongs", 
            "a fake computer-generated fork with impossible geometry"
        ]
    },
    "knife": {  # Fix the typo from "knive"
        "good_examples": [
            "a realistic knife with a distinct sharp blade and separate handle",
            "a knife showing a clear blade-handle junction and proper proportions",
            "a well-formed knife with straight blade edge and ergonomic handle"
        ],
        "ai_failures": [
            "a malformed knife with blended blade and handle",
            "a knife with wrong blade-to-handle proportions",
            "a distorted knife with bent blade or merged grip"
        ]
    },
    "bowl": {
        "good_examples": [
            "a properly shaped bowl with a smooth curved rim and consistent circular outline",
            "a realistic bowl with even, rounded edges and a uniform lip",
            "a bowl with a clear, unbroken circular rim"
        ],
        "ai_failures": [
            "a deformed bowl with jagged or broken rim",
            "a bowl with asymmetrical shape or uneven edges",
            "a bowl showing cracks or unnatural indentations"
        ]
    },
    "napkin": {
        "good_examples": [
            "a realistic napkin with proper fabric folds and rectangular shape",
            "a cloth napkin showing natural creases and soft edges",
            "a neatly folded napkin with visible texture and straight borders"
        ],
        "ai_failures": [
            "an artificial napkin with impossible geometry",
            "a napkin with strange textures or unnatural folds",
            "a napkin showing twisted corners and warped fabric"
        ]
    },
    "glass": {
        "good_examples": [
            "a transparent glass with smooth walls, clear rim, and realistic reflections",
            "a glass tumbler showing correct refraction and even walls",
            "a clear drinking glass with glossy highlights and proper thickness"
        ],
        "ai_failures": [
            "an opaque or distorted glass with impossible shape",
            "a glass showing irregular thickness or broken facets",
            "a glass with unrealistic reflections or warped geometry"
        ]
    },
    "knive": {  # Handle the typo in your data
        "good_examples": [
            "a realistic knife with a distinct sharp blade and separate handle",
            "a knife showing a clear blade-handle junction and proper proportions",
            "a well-formed knife with straight blade edge and ergonomic handle"
        ],
        "ai_failures": [
            "a malformed knife with blended blade and handle",
            "a knife with wrong blade-to-handle proportions",
            "a distorted knife with bent blade or merged grip"
        ]
    },
    "plate": {
        "good_examples": [
            "a round plate with smooth circular rim and flat even surface",
            "a plate showing a uniform edge and consistent curvature",
            "a realistic dinner plate with clear, unbroken rim"
        ],
        "ai_failures": [
            "a deformed plate with uneven or warp rim",
            "a plate with jagged edges or non-circular shape",
            "a plate showing cracks or wavy surface distortions"
        ]
    },
    "plate_dirt": {
        "good_examples": [
            "a round plate with realistic food stains and proper circular rim",
            "a plate showing natural-looking sauce or crumbs",
            "a dinner plate with slight, believable residue patterns"
        ],
        "ai_failures": [
            "a plate with artificial-looking residues or impossible stains",
            "a plate showing deformed shape and unnatural dirt patches",
            "a plate with smudges that float or form odd geometries"
        ]
    },
    "bowl_dirt": {
        "good_examples": [
            "a bowl with realistic food residues inside and proper curved rim",
            "a bowl showing natural drips or bits of food",
            "a soup bowl with believable soup stains and consistent shape"
        ],
        "ai_failures": [
            "a bowl with fake-looking residues or impossible stains",
            "a bowl with distorted shape and floating food bits",
            "a bowl showing unnatural color blobs and warped rim"
        ]
    }
}


def get_ai_quality_score(image_tensor, true_label, model):
    """
    Evaluate AI-generated object quality using contrastive analysis.
    Compares against specific AI failure patterns.
    Returns a score between 0 and 1, where 1 means realistic/well-formed.
    """
    if true_label not in AI_QUALITY_PROMPTS:
        print(f"Warning: Unknown label '{true_label}' - using default neutral score")
        return 0.5, 0.0, 0.0, "unknown_label"
    
    # Get contrastive prompts for this object type
    prompts     = AI_QUALITY_PROMPTS[true_label]
    good_list   = prompts["good_examples"]     
    fail_list   = prompts["ai_failures"]

    # Tokenize all prompts at once
    text_inputs = clip.tokenize(good_list + fail_list).to(device)
    
    with torch.no_grad():
        # Encode & normalize image
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Encode & normalize text
        text_features  = model.encode_text(text_inputs)
        text_features  = text_features / text_features.norm(dim=-1, keepdim=True) 

        # Compute cosine similarities against all prompts
        sims = (image_features @ text_features.T).squeeze(0)

        # Split into good vs failure sims
        n_good       = len(good_list)             
        good_sims    = sims[:n_good]              
        failure_sims = sims[n_good:]     

        # Average each group
        avg_good    = good_sims.mean().item() 
        avg_failure = failure_sims.mean().item()

        # Enhanced scoring method
        # 1. Direct difference approach
        quality_diff = avg_good - avg_failure
        
        # 2. Softmax approach (normalized)
        exp_g = torch.exp(torch.tensor(avg_good * 10))  # Scale up for more discrimination
        exp_f = torch.exp(torch.tensor(avg_failure * 10))
        quality_probability = (exp_g / (exp_g + exp_f)).item()
        
        # 3. Simple classification based on which is higher
        simple_classification = "good_example" if avg_good > avg_failure else "ai_failure"
        
        return quality_probability, avg_good, avg_failure, simple_classification

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

# Define quality thresholds - adjusted for better discrimination
HIGH_QUALITY_THRESHOLD = 0.6   # Clearly resembles good example  
MEDIUM_QUALITY_THRESHOLD = 0.4  # Somewhat ambiguous
# Anything below 0.4 is considered AI failure pattern

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
        quality_score, good_sim, failure_sim, classification = get_ai_quality_score(crop_input, true_label, model)
        
        # Categorize quality level
        if quality_score >= HIGH_QUALITY_THRESHOLD:
            quality_category = "good_example"
        elif quality_score >= MEDIUM_QUALITY_THRESHOLD:
            quality_category = "uncertain"
        else:
            quality_category = "ai_failure"

        # Calculate confidence and additional metrics
        confidence = abs(good_sim - failure_sim)
        similarity_ratio = good_sim / (failure_sim + 1e-8)  # Avoid division by zero
        
        result = {
            "image": img_path,
            "true_label": true_label,
            "assessment": classification,  # Direct classification from similarity comparison
            "quality_category": quality_category,  # Based on thresholds
            "quality_score": round(quality_score, 4),
            "good_similarity": round(good_sim, 4),
            "failure_similarity": round(failure_sim, 4),
            "confidence": round(confidence, 4),
            "similarity_ratio": round(similarity_ratio, 4)  # good/failure ratio
        }
        results.append(result)

        # Update per-image summary
        image_summary[img_path]["total"] += 1
        if quality_category == "good_example":
            image_summary[img_path]["high_quality_count"] += 1
        elif quality_category == "uncertain":
            image_summary[img_path]["medium_quality_count"] += 1
        else:
            image_summary[img_path]["low_quality_count"] += 1
            
        image_summary[img_path]["items"].append({
            "true_label": true_label,
            "assessment": classification,
            "quality_category": quality_category,
            "quality_score": round(quality_score, 4),
            "good_similarity": round(good_sim, 4),
            "failure_similarity": round(failure_sim, 4),
            "confidence": round(confidence, 4),
            "similarity_ratio": round(similarity_ratio, 4)
        })

        # More informative output
        print(f"{true_label}: {classification} (score: {quality_score:.3f}, confidence: {confidence:.3f}, ratio: {similarity_ratio:.2f})")

print(f"\n=== Analysis Summary ===")
total_objects = len(results)
good_examples = sum(1 for r in results if r["quality_category"] == "good_example")
uncertain = sum(1 for r in results if r["quality_category"] == "uncertain") 
ai_failures = sum(1 for r in results if r["quality_category"] == "ai_failure")

print(f"Total objects analyzed: {total_objects}")
print(f"Good examples: {good_examples} ({good_examples/total_objects*100:.1f}%)")
print(f"Uncertain: {uncertain} ({uncertain/total_objects*100:.1f}%)")
print(f"AI failures: {ai_failures} ({ai_failures/total_objects*100:.1f}%)")

# Analysis by direct classification (ignoring thresholds)
direct_good = sum(1 for r in results if r["assessment"] == "good_example")
direct_fail = sum(1 for r in results if r["assessment"] == "ai_failure")
print(f"\nDirect similarity comparison:")
print(f"Resembles good examples: {direct_good} ({direct_good/total_objects*100:.1f}%)")
print(f"Resembles AI failures: {direct_fail} ({direct_fail/total_objects*100:.1f}%)")

# Find most/least confident assessments
if results:
    most_confident = max(results, key=lambda x: x["confidence"])
    least_confident = min(results, key=lambda x: x["confidence"])
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    avg_ratio = sum(r["similarity_ratio"] for r in results) / len(results)
    
    print(f"\nAverage confidence: {avg_confidence:.3f}")
    print(f"Average good/failure ratio: {avg_ratio:.3f}")
    print(f"Most confident: {most_confident['true_label']} - {most_confident['assessment']} (confidence: {most_confident['confidence']:.3f})")
    print(f"Least confident: {least_confident['true_label']} - {least_confident['assessment']} (confidence: {least_confident['confidence']:.3f})")

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
