import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from networks.drbn import DRBN

project_root = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(project_root, "..", "data", "Raw", "low_val")
raw_output_dir = os.path.join(project_root, "..", "results", "DRBN")
model_path = os.path.join(project_root, "..", "checkpoints", "DRBN", "drbn.pth")

os.makedirs(raw_output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備： {device}")

model = DRBN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("模型載入完成")

# 建立將 PIL 圖片轉為 Tensor 的轉換器
to_tensor = transforms.ToTensor()

image_names = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
print(f"共找到 {len(image_names)} 張圖片")

for img_name in tqdm(image_names, desc="推論 DRBN"):
    try:
        img_path = os.path.join(input_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        input_tensor = to_tensor(image).unsqueeze(0).to(device)  # (B, C, H, W) 格式
        output_tensor = model(input_tensor).detach().cpu()

        # 儲存增強後結果，使用 normalize=True 讓圖像亮度落在 [0,1]
        save_path = os.path.join(raw_output_dir, img_name)
        save_image(output_tensor, save_path, normalize=True)

    except Exception as e:
        print(f"[錯誤] 處理 {img_name} 時發生問題：{e}")
print("所有圖片增強完成，已儲存至:", raw_output_dir)
