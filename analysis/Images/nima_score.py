import os
from PIL import Image
import torch
import torchvision.transforms as T
from torchmetrics.image import StructuralSimilarityIndexMeasure


device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize SSIM metric
ssim = StructuralSimilarityIndexMeasure().to(device)

# Image preprocessing: resize and convert to tensor
transform = T.Compose([
    T.Resize((256, 256)),  # or your desired size
    T.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)

ref_image_path = "C:/Users/shres/OneDrive/Desktop/NepWear/analysis/Images/real/Daura/053319_1.jpg"  
folder = "C:/Users/shres/OneDrive/Desktop/NepWear/analysis/Images/generated/Daura"

ref_img = load_image(ref_image_path)

ssim_scores = []

for f in os.listdir(folder):
    if f.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(folder, f)
        img = load_image(img_path)

        score = ssim(img, ref_img).item()
        ssim_scores.append(score)
        print(f"{f}: SSIM = {score:.4f}")

if ssim_scores:
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    print(f"\nAverage SSIM: {avg_ssim:.4f}")
else:
    print("No images found for evaluation.")
