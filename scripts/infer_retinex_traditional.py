import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from traditional.retinex_traditional import enhance_retinex  # 導入 Retinex 增強函式（已自定義）

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_folder = os.path.join(project_root, "data", "Raw", "low_val")
output_folder = os.path.join(project_root, "results", "RetinexTraditional")  # 儲存增強圖

os.makedirs(output_folder, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),  # 將 PIL 圖片轉為 tensor 格式，範圍 [0, 1]
])

device = torch.device("cpu")

image_list = sorted(os.listdir(input_folder))
print(f"開始 RetinexTraditional 推論，共 {len(image_list)} 張圖像...")

for img_name in tqdm(image_list, desc="推論 RetinexTraditional"):
    img_path = os.path.join(input_folder, img_name)
    try:
        # 開啟並轉換為 RGB 模式
        img = Image.open(img_path).convert("RGB")

        # 不需計算梯度（推論模式）
        with torch.no_grad():
            tensor = transform(img).unsqueeze(0).to(device)  # 增加 batch 維度
            enhanced_tensor = enhance_retinex(tensor)        # 執行 Retinex 增強
            enhanced_tensor = torch.clamp(enhanced_tensor, 0, 1)  # 限制結果值於 [0, 1]

            # 將增強結果轉回 PIL 圖片
            enhanced_np = (enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            enhanced_img = Image.fromarray(enhanced_np)

            base_name = os.path.splitext(img_name)[0] + ".png"
            save_path = os.path.join(output_folder, base_name)
            enhanced_img.save(save_path)

    except Exception as e:
        print(f"[Error] Failed to process {img_path}: {e}")
