# 📁 Step 2: Import Libraries
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import gradio as gr

# ⚙️ Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# 🔮 Main Inference Function
def generate_realistic_image(input_image):
    # Generate Caption
    inputs = blip_processor(images=input_image, return_tensors="pt").to(device)
    caption_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)

    # Refine Prompt
    prompt = f"a realistic photo of {caption}"

    # Generate Image
    with torch.autocast(device.type) if device.type == "cuda" else torch.no_grad():
        result = sd_pipe(prompt).images[0]

    return caption, result

# 🌐 Gradio Interface
title = "🎨 Doodle to Realistic Image Generator"
description = "Upload a doodle or sketch and let AI transform it into a realistic image using BLIP and Stable Diffusion"

demo = gr.Interface(
    fn=generate_realistic_image,
    inputs=gr.Image(type="pil", label="Upload your doodle"),
    outputs=[
        gr.Textbox(label="Generated Caption"),
        gr.Image(type="pil", label="Realistic Image Output")
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

# 🚀 Launch App
demo.launch()
