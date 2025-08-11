import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasetNep.dataset import NepVTONDataset
from pipelinenep import NepaliWearVtonPipeline  # Change to your actual model file

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset and DataLoader
dataset = NepVTONDataset(data_root=".", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Load pre-trained model
# model = NepaliWearVtonPipeline.from_pretrained("checkpoints/ootd", torch_dtype=torch.float32)
model = NepaliWearVtonPipeline.from_pretrained(
    "C:/Users/shres/OneDrive/Desktop/NepWear/checkpoints/ootd", 
    local_files_only=True, 
    torch_dtype=torch.float32
)

model.to(device)
model.train()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training Loop
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for batch in tqdm(dataloader):
        person = batch["person"].to(device)
        cloth = batch["cloth"].to(device)

        # Forward pass
        outputs = model(person_image=person, cloth_image=cloth)

        loss = outputs["loss"]  # Your model must return a loss value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")

# Save checkpoint
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "../checkpoints/nepvton_finetuned.pth")
