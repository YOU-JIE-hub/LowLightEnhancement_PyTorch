import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from networks.retinexnet import UNetDecomNet, EnhanceNetDeep

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_folder   = os.path.join(project_root, "data", "Raw", "low_val")
output_folder  = os.path.join(project_root, "results", "RetinexNet")
decom_ckpt     = os.path.join(project_root, "checkpoints", "RetinexNet", "RetinexNet_decom.pth")
enhance_ckpt   = os.path.join(project_root, "checkpoints", "RetinexNet", "RetinexNet_enhance.pth")

os.makedirs(output_folder, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

decom_net = UNetDecomNet().to(device)
enhance_net = EnhanceNetDeep().to(device)

decom_net.load_state_dict(torch.load(decom_ckpt, map_location=device))
enhance_net.load_state_dict(torch.load(enhance_ckpt, map_location=device))

decom_net.eval()
enhance_net.eval()

transform = transforms.ToTensor()

image_list = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'png'))])

for img_name in tqdm(image_list, desc="推論 RetinexNet"):
    img_path = os.path.join(input_folder, img_name)
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        R, I = decom_net(img_tensor)                     # 1. DecomNet 分解 R, I
        I_new, color_map = enhance_net(R, I)              # 2. EnhanceNet 同時輸出 I_new 和 color_map
        enhanced = (R + color_map) * I_new.expand_as(R)   # 3. 合成增強圖
        enhanced = torch.clamp(enhanced, 0, 1)            # 4. 限制到合法範圍

    enhanced_img = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_img = (enhanced_img * 255).clip(0, 255).astype('uint8')
    enhanced_img = Image.fromarray(enhanced_img)

    enhanced_img.save(os.path.join(output_folder, img_name))
print("所有圖片增強完成，已儲存至:", output_folder)
