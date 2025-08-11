
import gradio as gr
import os
from pathlib import Path
import sys
import torch
from PIL import Image
import numpy as np

from utilsnep import get_mask_location
from analysis.Images.full_visualisation import run_evaluation

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from inference.inference_Nepdc import NepWear

# Initialize models
openpose_model_dc = OpenPose(0)
parsing_model_dc = Parsing(0)
NepwearDc = NepWear(0)

# Constants
category_utils = 'dresses'
example_path = os.path.join(os.path.dirname(__file__), 'images')
model_path = os.path.join(example_path, 'model/model_9.png')
default_garment_path = os.path.join(example_path, 'clothes/tamang.png')

# Garment metadata
garment_metadata = {
    "gurung.png": (
        "Gurung Dress",
        "Western and Central Nepal",
        "A traditional women's outfit featuring a richly embroidered velvet blouse and a pleated wraparound skirt, worn with ornate gold jewelry and a beaded necklace during festivals like Tihar and Lhosar."
    ),
    "hakku.png": (
        "Haku Patasi",
        "Kathmandu Valley (Newar Community)",
        "A traditional black sari with red borders, paired with a matching blouse and shawl, worn by Newar women during cultural events and rituals."
    ),
    "bakhu.jpg": (
        "Bhaku with Honju",
        "Himalayan Regions (Sherpa/Tibetan Communities)",
        "A long sleeveless wraparound dress (Bhaku) worn over a full-sleeved blouse (Honju), often with an apron and jewelry; traditional to Tibetan-Nepali communities."
    ),
    "daura.jpg": (
        "Daura Suruwal",
        "All Nepal",
        "The national dress for Nepali men, featuring a double-breasted, closed-neck shirt with ties (Daura) and tapered trousers (Suruwal); now adapted in modern styles for both genders."
    ),
    "kurtha.png": (
        "Kurta Suruwal with Dupatta",
        "Madhesh/Terai Region",
        "A three-piece outfit consisting of a tunic (Kurta), tapered pants (Suruwal), and a draped scarf (Dupatta); commonly worn by women for daily and formal occasions."
    ),
    "tamang.png": (
        "Tamang Traditional Dress",
        "Hilly Regions (Tamang Community)",
        "A vibrant outfit including a wrap skirt (Guniu), short blouse, and colorful sash, adorned with bead necklaces and silver jewelry, worn by Tamang women during cultural festivals like Sonam Lhosar."
    )
}

example_garments = [os.path.join(example_path, 'clothes', fname) for fname in garment_metadata]

def process_dress(vton_img, garm_img, n_samples, n_steps, image_scale, seed):
    with torch.no_grad():
        garm_img = Image.open(garm_img).resize((768, 1024))
        vton_img = Image.open(vton_img).resize((768, 1024))
        keypoints = openpose_model_dc(vton_img.resize((384, 512)))
        model_parse, _ = parsing_model_dc(vton_img.resize((384, 512)))

        mask, mask_gray = get_mask_location('dc', category_utils, model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        images = NepwearDc(
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

def show_if_example(filepath):
    if filepath is None or not os.path.isfile(filepath):
        return "### "  # empty if no file or file missing

    filename = os.path.basename(filepath).lower().replace(".jpeg", ".jpg")
    print("Selected garment:", filename)  # debug print

    if filename in garment_metadata:
        title, origin, desc = garment_metadata[filename]
        return f"### {title}\n📍 **Origin:** {origin}\n\n📖 {desc}"
    else:
        return "### "  # empty if not in examples

with gr.Blocks() as demo:
    tabs = gr.Tabs()

    with tabs:
        with gr.TabItem("Virtual Try-On"):
            with gr.Row():
                gr.Markdown("## 🧵 NepWear – Try On Traditional Nepali Garments")
            with gr.Row():
                # Left Column - Garment Information
                with gr.Column(scale=1):
                    garment_info = gr.Markdown("### Garment info will appear here")
                
                # Middle Column - Garment Selection
                with gr.Column(scale=1):
                    garm_img = gr.Image(label="Selected Garment", sources='upload', type="filepath", height=384, value=default_garment_path)
                    gr.Examples(
                        label="Choose Traditional Garment",
                        inputs=garm_img,
                        examples=example_garments,
                        examples_per_page=6
                    )
                
                # Right Column - Model Image
                with gr.Column(scale=1):
                    vton_img = gr.Image(label="Upload Model Image", sources='upload', type="filepath", height=384, value=model_path)
                    gr.Examples(
                        label="Model Examples",
                        inputs=vton_img,
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
                            os.path.join(example_path, 'model/model_6.png'),
                        ],
                        examples_per_page=6
                    )

            # Results Section Below
            with gr.Row():
                result_gallery = gr.Gallery(label='Virtual Try-On Results', show_label=True, elem_id="gallery", preview=True, columns=4)

            # Controls Section
            with gr.Row():
                with gr.Column():
                    run_button = gr.Button(value="Generate Try-On", variant="primary")
                with gr.Column():
                    n_samples = gr.Slider(label="Number of Output Images", minimum=1, maximum=4, value=1, step=1)
                    n_steps = gr.Slider(label="Generation Steps", minimum=1, maximum=40, value=5, step=1)
                    image_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
                    seed = gr.Slider(label="Random Seed (-1 for random)", minimum=-1, maximum=2147483647, step=1, value=-1)

            # Trigger info update whenever garment image changes
            garm_img.change(fn=show_if_example, inputs=garm_img, outputs=garment_info)

            run_button.click(
                fn=process_dress,
                inputs=[vton_img, garm_img, n_samples, n_steps, image_scale, seed],
                outputs=[result_gallery]
            )

        

        with gr.TabItem("Dashboard"):
            gr.Markdown("## 📊 NepWear Evaluation")

            with gr.Row():
                eval_button = gr.Button("Run Realism Evaluation")

            with gr.Row():
                eval_plot = gr.Image(label="Evaluation Plot")

            with gr.Row():
                eval_text = gr.Markdown()

            def wrapper_run_eval():
                img, results = run_evaluation()
                summary = "### Realism Rankings\n"
                for name, val in sorted(results.items(), key=lambda x: -x[1]["Realism Score"]):
                    summary += f"- **{name}**: Realism Score {val['Realism Score']:.2f}/6\n"
                overall = np.mean([v["Realism Score"] for v in results.values()])
                summary += f"\n**Overall Average Realism Score:** {overall:.2f}/6"
                return img, summary

            eval_button.click(fn=wrapper_run_eval, inputs=[], outputs=[eval_plot, eval_text])



    demo.launch(server_name='127.0.0.1', server_port=7865)
