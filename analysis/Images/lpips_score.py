
import lpips
import torch
from PIL import Image
from torchvision import transforms
import os
import csv

# -----------------------------
# LPIPS Model Initialization
# -----------------------------
loss_fn = lpips.LPIPS(net='alex')  # 'vgg' or 'squeeze' also valid

# -----------------------------
# Preprocessing Function
# -----------------------------
to_tensor = transforms.ToTensor()

def load_and_prepare(image_path):
    image = Image.open(image_path).convert("RGB").resize((256, 256))
    tensor = to_tensor(image).unsqueeze(0) * 2 - 1  # Normalize to [-1, 1]
    return tensor

# -----------------------------
# Input Paths
# -----------------------------
ref_image_path = "C:/Users/shres/OneDrive/Desktop/NepWear/analysis/Images/real/Daura/053319_1.jpg"
gen_folder = "C:/Users/shres/OneDrive/Desktop/NepWear/analysis/Images/generated/Daura"

# Load reference image once
ref_tensor = load_and_prepare(ref_image_path)

# -----------------------------
# Compare to Each Image in Folder
# -----------------------------
gen_images = sorted([f for f in os.listdir(gen_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

lpips_scores = []

for gen_img in gen_images:
    gen_path = os.path.join(gen_folder, gen_img)
    gen_tensor = load_and_prepare(gen_path)

    score = loss_fn(ref_tensor, gen_tensor).item()
    print(f"{gen_img} vs ref_image -> LPIPS: {score:.4f}")
    lpips_scores.append((gen_img, score))

# -----------------------------
# Output Average and Save to CSV
# -----------------------------
average_score = sum(score for _, score in lpips_scores) / len(lpips_scores)
print(f"\n✅ Average LPIPS Score: {average_score:.4f}")

with open("lpips_scores_single_vs_folder.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["generated_image", "lpips_score"])
    for gen_img, score in lpips_scores:
        writer.writerow([gen_img, f"{score:.4f}"])

print("📁 Results saved to lpips_scores_single_vs_folder.csv")
