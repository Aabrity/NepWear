
import os
import csv
import torch
import clip
from PIL import Image

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Folder with generated images
folder = "generated/Daura"

# Prompt to compare against
text_prompt = "An iconic regional suit, comprising a flowing top with unique closures and close-fitting pants"
text = clip.tokenize([text_prompt]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)

scores = []

for file in os.listdir(folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        image_path = os.path.join(folder, file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
            score = similarity.item()
            scores.append((file, score))

if scores:
    # Write to CSV
    csv_path = "clip_scores.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "clip_similarity_score"])
        writer.writerows(scores)

    print(f"Scores saved to {csv_path}")
    average_score = sum(score for _, score in scores) / len(scores)
    print(f"\nAverage CLIP Score: {average_score:.4f}")
else:
    print("No images found in the folder.")
