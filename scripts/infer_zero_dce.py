import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from networks.zero_dce import ZeroDCE
from utils.dce_utils import apply_curve              # 將網路輸出 A 應用於原始圖像曲線上
from utils.postprocess import postprocess_gamma      # 後處理函式，進行 Gamma + Gain 提亮

input_folder = r'C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\data\Raw\low_val'  # 輸入資料夾（低光圖）
raw_output_folder = r'C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results\ZeroDCE_val'  # 增強結果（原始）
post_output_folder = os.path.join(raw_output_folder, 'post')  # 增強結果（後處理版本）
model_path = r'C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\checkpoints\ZeroDCE\zero_dce.pth'  # 模型權重

os.makedirs(raw_output_folder, exist_ok=True)
os.makedirs(post_output_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ZeroDCE().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()                                                           #

transform = transforms.Compose([transforms.ToTensor()])

img_list = sorted(os.listdir(input_folder))
print(f"開始 Zero-DCE 推論，共 {len(img_list)} 張圖像...")

for img_name in tqdm(img_list):
    img_path = os.path.join(input_folder, img_name)

    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':              # 若非 RGB，轉為 RGB 模式
            img = img.convert('RGB')

        img_tensor = transform(img).unsqueeze(0).to(device)  # 圖片轉為 [1, 3, H, W] 並放到裝置上

        with torch.no_grad():              # 關閉梯度以節省記憶體
            A = model(img_tensor)          # 模型輸出為調整曲線參數 A
            enhanced = apply_curve(img_tensor, A)  # 將 A 應用於原圖以產生增強影像

        base_name = os.path.splitext(img_name)[0] + ".png"
        post_name = os.path.splitext(img_name)[0] + "_post.png"

        enhanced_np = enhanced.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255
        enhanced_np = enhanced_np.astype(np.uint8)
        raw_img = Image.fromarray(enhanced_np)
        raw_img.save(os.path.join(raw_output_folder, base_name))

        # === 儲存後處理版本（post）===
        post_img = postprocess_gamma(raw_img, gamma=0.5, gain=1.4)
        post_img.save(os.path.join(post_output_folder, post_name))

    except Exception as e:
        print(f"無法處理 {img_path}：{e}")
