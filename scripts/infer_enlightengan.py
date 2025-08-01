import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from networks.enlightengan_unet import ImprovedUNet

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt_path = os.path.join(project_root, "checkpoints", "EnlightenGAN", "EnlightenGAN.pth")
input_dir  = os.path.join(project_root, "data", "Raw", "low_val")
save_dir   = os.path.join(project_root, "results", "EnlightenGAN")
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備： {device}")
model = ImprovedUNet().to(device)
print("模型載入完成")

# 加入 fallback：支援含 "G"、"G_state"、或純 state_dict
checkpoint = torch.load(ckpt_path, map_location=device)
if isinstance(checkpoint, dict):
    if "G" in checkpoint:
        model.load_state_dict(checkpoint["G"])
    elif "G_state" in checkpoint:
        model.load_state_dict(checkpoint["G_state"])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.eval()

transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.pow(0.4545))  # gamma 還原
])

for name in tqdm(os.listdir(input_dir), desc="推論 EnlightenGAN"):
    if not name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    path = os.path.join(input_dir, name)
    image = Image.open(path).convert("RGB")
    orig_size = image.size  # (W, H)

    input_tensor = transforms.Resize((256, 256))(image)
    input_tensor = transform_tensor(input_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.clamp(output, 0, 1)
        output = torch.nn.functional.interpolate(output, size=orig_size[::-1], mode="bilinear", align_corners=False)

    save_image(output, os.path.join(save_dir, name))

print("所有圖片增強完成，已儲存至:", save_dir)
