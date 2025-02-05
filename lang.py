from datetime import datetime
current_time = datetime.now()
import torch
from PIL import Image
from lang_sam import LangSAM
import numpy as np

model = LangSAM()
image_pil = Image.open("/home/rbccps/zero_shot/diadem_sample.jpeg").convert("RGB")
text_prompt = "purple car"
results = model.predict([image_pil], [text_prompt])
mask_len = results[0]['masks']
mask = mask_len.squeeze()
large_array = np.random.rand(720, 1280)
y_indices, x_indices = np.where(mask == 1)
pixel_locations = np.column_stack((x_indices, y_indices))
num_samples = min(20, len(pixel_locations))  # Ensure we don't exceed available pixels
sampled_pixels = pixel_locations[np.random.choice(len(pixel_locations), num_samples, replace=False)]
print(sampled_pixels)  # Print sampled (x, y) locations
print("2D Keypoint extraction successfull !!")
duration = datetime.now() - current_time
print(f"Duration: {duration}") # 11.52 seconds