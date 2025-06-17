import os
from tqdm import tqdm
from traditional.enhance_freq_filter import enhance_freq_filter
from utils.postprocess import postprocess_gamma

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_dir = os.path.join(project_root, "data", "Raw", "low_val")
output_dir_raw = os.path.join(project_root, "results", "FreqFilter")
output_dir_post = os.path.join(output_dir_raw, "post")

os.makedirs(output_dir_raw, exist_ok=True)
os.makedirs(output_dir_post, exist_ok=True)

image_list = sorted([
    f for f in os.listdir(input_dir)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

print(f"開始 FreqFilter 頻域增強，共 {len(image_list)} 張圖像...")

for filename in tqdm(image_list, desc="推論 FreqFilter"):
    input_path = os.path.join(input_dir, filename)
    try:
        raw_img = enhance_freq_filter(input_path)
        post_img = postprocess_gamma(raw_img, gamma=0.5, gain=1.4)

        base_name = os.path.splitext(filename)[0]
        raw_path = os.path.join(output_dir_raw, f"{base_name}.png")
        post_path = os.path.join(output_dir_post, f"{base_name}_post.png")

        raw_img.save(raw_path)
        post_img.save(post_path)

    except Exception as e:
        print(f"[Error] Failed to process {input_path}: {e}")
