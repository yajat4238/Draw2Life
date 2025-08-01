# 📁 Step 2: Import Libraries
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
 
# 🧠 Step 3: Load BLIP (Image Captioning Model)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
 
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
