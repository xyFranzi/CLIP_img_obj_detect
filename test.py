import torch
import clip
import numpy as np
from PIL import Image
import json
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CLIP available models: {clip.available_models()}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test loading CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("CLIP model loaded successfully!")