import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from networks.enlightengan_unet import ImprovedUNet

model_path = r"C:/Users/hurry/OneDrive/桌面/LowLightEnhancement_PyTorch/checkpoints/EnlightenGAN/EnlightenGAN_100.pth"
input_dir  = r"C:/Users/hurry/OneDrive/桌面/LowLightEnhancement_PyTorch/data/Raw/low_val"
save_dir  = r"C:/Users/hurry/OneDrive/桌面/LowLightEnhancement_PyTorch/results/EnlightenGAN_val"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedUNet().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["G"])
model.eval()

transform_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.pow(0.4545))  # gamma 還原
])

for name in tqdm(os.listdir(input_dir), desc="🔍 推論中"):
    if not name.lower().endswith((".jpg", ".png")):
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

print(f"\nEnlightenGAN 推論完成，結果已還原原始解析度：{save_dir}")
