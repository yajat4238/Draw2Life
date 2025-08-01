
# 📁 Step 2: Import Libraries
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
 
# 🧠 Step 3: Load BLIP (Image Captioning Model)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
 
# 🎨 Step 4: Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
 
# 🖼️ Step 5: Load and Display the Sketch
image_path = "/content/5223503110340608.png"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
 
plt.imshow(image)
plt.axis("off")
plt.title("Input Sketch")
plt.show()
 
# ✏️ Step 6: Generate Caption Using BLIP
inputs = blip_processor(images=image, return_tensors="pt").to("cuda")
caption_ids = blip_model.generate(**inputs)
caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
print("Generated Caption:", caption)
 
# 📝 Step 7: Refine Caption for Realistic Image Generation
prompt = f"a realistic photo of {caption}"
 
# 🖌️ Step 8: Generate Realistic Image
with torch.autocast("cuda"):
    result = pipe(prompt).images[0]
 
# Display the result
plt.imshow(result)
plt.axis("off")
plt.title(f"Generated Image: {prompt}")
plt.show()
 
# 💾 Step 9: Save the Output
result.save("generated_image.png")



# 🖥️ Streamlit App: doodle_to_realistic.py
import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

# 🎬 App Title
st.set_page_config(page_title="Doodle to Realistic Image", layout="centered")
st.title("🎨 Doodle to Realistic Image Generator")
st.markdown("Upload a sketch or doodle to transform it into a realistic image using AI magic!")

# ⚙️ Load Models (cached for performance)
@st.cache_resource
def load_models():
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to("cuda")
    return blip_processor, blip_model, sd_pipe

blip_processor, blip_model, sd_pipe = load_models()

# 📤 Upload Image
uploaded_file = st.file_uploader("Upload your doodle (PNG or JPG)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Uploaded Sketch", use_column_width=True)

    # ✏️ Generate Caption
    with st.spinner("Generating caption..."):
        inputs = blip_processor(images=input_image, return_tensors="pt").to("cuda")
        caption_ids = blip_model.generate(**inputs)
        caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
        st.success(f"🧠 Caption: {caption}")

    # 🎨 Refine Prompt
    prompt = f"a realistic photo of {caption}"
    st.info(f"🎯 Prompt for Stable Diffusion: `{prompt}`")

    # 🖌️ Generate Realistic Image
    if st.button("Generate Realistic Image"):
        with st.spinner("Creating your masterpiece..."):
            with torch.autocast("cuda"):
                result = sd_pipe(prompt).images[0]
            st.image(result, caption="Realistic Output", use_column_width=True)
            result.save("generated_image.png")
            st.success("✅ Image Saved as `generated_image.png`")

# 🧾 Footer
st.markdown("---")
st.markdown("Made with 🤖 using BLIP + Stable Diffusion")
