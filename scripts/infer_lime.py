import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from traditional.lime_enhance import enhance_lime

input_folder = r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\data\Raw\low_val"
output_folder = r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results\LIME_val"

os.makedirs(output_folder, exist_ok=True)

image_list = sorted(os.listdir(input_folder))
print(f"開始 LIME 推論，共 {len(image_list)} 張圖像...")

for img_name in tqdm(image_list):
    img_path = os.path.join(input_folder, img_name)
    try:
        img = Image.open(img_path).convert('RGB')               # 讀取並轉為 RGB 格式
        enhanced = enhance_lime(np.asarray(img))                # 呼叫 LIME 增強方法進行處理

        if isinstance(enhanced, tuple):                         # 若返回的是 (結果圖, illumination map)
            enhanced = enhanced[0]                              # 僅取第一張為增強結果

        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)   # 限制像素值在 0~255 並轉為整數
        enhanced_img = Image.fromarray(enhanced)                # 轉回 PIL 影像

        base_name = os.path.splitext(img_name)[0] + ".png"
        save_path = os.path.join(output_folder, base_name)
        enhanced_img.save(save_path)

    except Exception as e:
        print(f"[Error] Failed to process {img_path}: {e}")
