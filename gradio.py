import gradio as gr
import os
from pathlib import Path
import sys
import torch
from PIL import Image

from utils_ootd import get_mask_location

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from inference.inference_Nepdc import NepWear

openpose_model_dc = OpenPose(0)
parsing_model_dc = Parsing(0)
ootd_model_dc = NepWear(0)

# Dress-only constants
category = 2  # dress
category_str = 'dress'
category_utils = 'dresses'

example_path = os.path.join(os.path.dirname(__file__), 'images')
model_path = os.path.join(example_path, 'model/model_9.png')
garment_path = os.path.join(example_path, 'clothes/053290_1.jpg')

def process_dress(vton_img, garm_img, n_samples, n_steps, image_scale, seed):
    model_type = 'dc'
    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))
        keypoints = openpose_model_dc(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_dc(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location(model_type, category_utils, model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
        
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = ootd_model_dc(
            model_type=model_type,
            category=category_str,
            image_garm=garm_img,
            image_vton=masked_vton_img,
            mask=mask,
            image_ori=vton_img,
            num_samples=n_samples,
            num_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
        )

    return images

# Gradio UI
block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## NepWear")

    with gr.Row():
        with gr.Column():
            vton_img = gr.Image(label="Model Image", sources='upload', type="filepath", height=384, value=model_path)
            gr.Examples(
                inputs=vton_img,
                examples_per_page=6,
                examples=[
                    os.path.join(example_path, 'model/model_9.png'),
                    os.path.join(example_path, 'model/model_5.png'),
                    os.path.join(example_path, 'model/model_7.png'),
                    os.path.join(example_path, 'model/model_8.png'),
                    os.path.join(example_path, 'model/052767_0.jpg'),
                    os.path.join(example_path, 'model/052472_0.jpg'),
                    os.path.join(example_path, 'model/053514_0.jpg'),
                    os.path.join(example_path, 'model/049205_0.jpg'),
                    os.path.join(example_path, 'model/049713_0.jpg'),
                    os.path.join(example_path, 'model/051918_0.jpg'),
                    os.path.join(example_path, 'model/051962_0.jpg'),
                    os.path.join(example_path, 'model/model_6.jpg'),
                ])

        with gr.Column():
            garm_img = gr.Image(label="Dress Garment", sources='upload', type="filepath", height=384, value=garment_path)
            gr.Examples(
                inputs=garm_img,
                examples_per_page=6,
                examples=[
                    os.path.join(example_path, 'clothes/051998_1.jpg'),
                    os.path.join(example_path, 'clothes/052234_1.jpg'),
                    os.path.join(example_path, 'clothes/053290_1.jpg'),
                    os.path.join(example_path, 'clothes/053319_1.jpg'),
                    os.path.join(example_path, 'clothes/053786_1.jpg'),
                    # os.path.join(example_path, 'clothes/053319_1.jpg'),
                ])

        with gr.Column():
            result_gallery = gr.Gallery(label='Result', show_label=False, elem_id="gallery", preview=True, scale=1)

    with gr.Column():
        run_button = gr.Button(value="Run Try-On")
        n_samples = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
        n_steps = gr.Slider(label="Steps", minimum=20, maximum=40, value=20, step=1)
        image_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)

    inputs = [vton_img, garm_img, n_samples, n_steps, image_scale, seed]
    run_button.click(fn=process_dress, inputs=inputs, outputs=[result_gallery])

block.launch(server_name='127.0.0.1', server_port=7865)
