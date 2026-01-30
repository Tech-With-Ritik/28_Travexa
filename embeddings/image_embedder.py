from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

TARGET_DIM = 384

def embed_image(path):
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        emb = model.get_image_features(**inputs)[0].cpu().numpy()

    return emb[:TARGET_DIM].tolist()
