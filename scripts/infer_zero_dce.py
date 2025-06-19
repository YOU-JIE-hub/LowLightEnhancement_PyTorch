import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from networks.zero_dce import ZeroDCE
from utils.dce_utils import apply_curve              # 將網路輸出 A 應用於原始圖像曲線上
from utils.postprocess import postprocess_gamma      # 後處理函式，進行 Gamma + Gain 提亮

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_folder = os.path.join(project_root, "data", "Raw", "low_val")
raw_output_folder = os.path.join(project_root, "results", "ZeroDCE")
post_output_folder = os.path.join(raw_output_folder, "post")
model_path = os.path.join(project_root, "checkpoints", "ZeroDCE", "zero_dce.pth")

os.makedirs(raw_output_folder, exist_ok=True)
os.makedirs(post_output_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備： {device}")

model = ZeroDCE().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("模型載入完成")

transform = transforms.Compose([transforms.ToTensor()])

img_list = sorted(os.listdir(input_folder))
print(f"開始 Zero-DCE 推論，共 {len(img_list)} 張圖像...")

for img_name in tqdm(img_list, desc="推論 Zero-DCE"):
    img_path = os.path.join(input_folder, img_name)

    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            A = model(img_tensor)
            enhanced = apply_curve(img_tensor, A)

        base_name = os.path.splitext(img_name)[0] + ".png"
        post_name = os.path.splitext(img_name)[0] + "_post.png"

        enhanced_np = enhanced.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255
        enhanced_np = enhanced_np.astype(np.uint8)
        raw_img = Image.fromarray(enhanced_np)
        raw_img.save(os.path.join(raw_output_folder, base_name))

        post_img = postprocess_gamma(raw_img, gamma=0.5, gain=1.4)
        post_img.save(os.path.join(post_output_folder, post_name))

    except Exception as e:
        print(f"無法處理 {img_path}：{e}")
print("所有圖片增強完成，已儲存至:", raw_output_folder)
print("後處理補光的圖片，已儲存至:", post_output_folder)
