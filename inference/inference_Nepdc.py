import torch
import torch.nn.functional as F
import random
import time
from pathlib import Path
import sys

from PIL import Image
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from transformers import AutoProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTokenizer

from pipeline.py import NepaliWearVtonPipeline
from models.unet_Nepgarm_condition import UNetGarm2DConditionModel
from models.unet_Nepvton_condition import UNetVton2DConditionModel

# Set up project path
PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Checkpoint paths (adjust if needed)
VIT_PATH = "../checkpoints/clip-vit-large-patch14"
VAE_PATH = "../checkpoints/ootd"
UNET_PATH = "../checkpoints/ootd/ootd_dc/checkpoint-36000"
MODEL_PATH = "../checkpoints/ootd"


class NepWear:
    def __init__(self, gpu_id=0):
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        # Load VAE, UNets
        vae = AutoencoderKL.from_pretrained(VAE_PATH, subfolder="vae", torch_dtype=torch.float16)
        unet_garm = UNetGarm2DConditionModel.from_pretrained(UNET_PATH, subfolder="unet_garm", torch_dtype=torch.float16, use_safetensors=True)
        unet_vton = UNetVton2DConditionModel.from_pretrained(UNET_PATH, subfolder="unet_vton", torch_dtype=torch.float16, use_safetensors=True)

        # Build pipeline
        self.pipe = NepaliWearVtonPipeline.from_pretrained(
            MODEL_PATH,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        # Replace scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        # Load encoders
        self.processor = AutoProcessor.from_pretrained(VIT_PATH)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder").to(self.device)

    def _tokenize_caption(self, text, max_length=3):
        tokens = self.tokenizer(text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        return tokens.input_ids.to(self.device)

    def __call__(self, image_garm, image_vton, mask, image_ori, num_samples=1, num_steps=20, image_scale=1.0, seed=None):
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        generator = torch.manual_seed(seed)
        print(f"Using seed: {seed}")

        with torch.no_grad():
            # Encode garment image
            garm_input = self.processor(images=image_garm, return_tensors="pt").to(self.device)
            garm_emb = self.image_encoder(garm_input["pixel_values"]).image_embeds.unsqueeze(1)

            # Encode "dress" category
            text_emb = self.text_encoder(self._tokenize_caption(["dress"]))[0]
            prompt_embeds = torch.cat([text_emb, garm_emb], dim=1)

            # Run pipeline
            result = self.pipe(
                prompt_embeds=prompt_embeds,
                image_garm=image_garm,
                image_vton=image_vton,
                mask=mask,
                image_ori=image_ori,
                num_inference_steps=num_steps,
                image_guidance_scale=image_scale,
                num_images_per_prompt=num_samples,
                generator=generator,
            )

        return result.images
