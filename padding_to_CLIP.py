import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
import datetime

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'data')
images_dir = os.path.join(data_dir, 'img')
bboxes_dir = os.path.join(data_dir, 'box')

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Define prompt templates for each class
CLASSES = {
    "fork": ["a photo of a fork"],
    "bowl": ["a photo of a bowl"],
    "napkin": ["a photo of a napkin"],
    "glass": ["a photo of a glass"],
    "knive": ["a photo of a knife"],  # typo in annotation
    "plate": ["a photo of a plate"],
    "plate_dirt": ["a plate with food residues"],
    "bowl_dirt": ["a bowl with food residues"]
}

# Preprocess class text prompts
text_tokens = []
class_names = []
for class_name, prompts in CLASSES.items():
    for prompt in prompts:
        text_tokens.append(clip.tokenize(prompt))
        class_names.append(class_name)

text_tokens = torch.cat(text_tokens).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

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

for img_path, bbox_path in tqdm(imagedirmap.items()):
    full_img_path = os.path.join(current_dir, img_path)
    full_bbox_path = os.path.join(current_dir, bbox_path)

    image = Image.open(full_img_path).convert("RGB")
    width, height = image.size

    with open(full_bbox_path, 'r') as f:
        boxes = json.load(f)

    for box in boxes:
        label = box["name"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        # Apply padding
        pad = 15  # pixels
        x1_p, y1_p, x2_p, y2_p = apply_padding(x1, y1, x2, y2, pad, width, height)

        # Crop and preprocess
        crop = image.crop((x1_p, y1_p, x2_p, y2_p))
        crop_input = preprocess(crop).unsqueeze(0).to(device)

        # CLIP inference
        with torch.no_grad():
            image_feature = model.encode_image(crop_input)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            similarities = (image_feature @ text_features.T).squeeze(0)

        best_idx = similarities.argmax().item()
        predicted_label = class_names[best_idx]
        score = similarities[best_idx].item()

        save_dir = "cropped_regions"
        os.makedirs(save_dir, exist_ok=True)
        crop.save(os.path.join(save_dir, f"{label}_{predicted_label}_{x1}_{y1}.png"))

        result = {
            "image": img_path,
            "true_label": label,
            "predicted_label": predicted_label,
            "match": predicted_label == label,
            "similarity": score,
        }
        results.append(result)

        print(result)  # or store/save later
