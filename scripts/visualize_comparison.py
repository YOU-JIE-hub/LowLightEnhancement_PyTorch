import os
import random
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft JhengHei'
from utils.postprocess import postprocess_gamma

low_dir = r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\data\Raw\low"
gt_dir = r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\data\Raw\high"
output_dir = r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results\Comparison"
os.makedirs(output_dir, exist_ok=True)

model_configs = {
    "Low":               (low_dir, False),
    "GT":                (gt_dir, False),
    "FreqFilter":        (r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results\FreqFilter\post", True),
    "RetinexNet":        (r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results\RetinexNet", False),
    "ZeroDCE":           (r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results\ZeroDCE\post", True),
    "RetinexTraditional":(r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results\RetinexTraditional", False),
    "LIME":              (r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results\LIME", False),
    "EnlightenGAN":      (r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results\EnlightenGAN", False),
    "DRBN":              (r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results\DRBN", False),
}

sample_names = random.sample(
    [f for f in os.listdir(gt_dir) if f.endswith(".png")], 5
)

for name in sample_names:
    name_base = os.path.splitext(name)[0]
    fig, axs = plt.subplots(len(model_configs), 1, figsize=(6, len(model_configs) * 3))  # 每張圖一行
    if len(model_configs) == 1:
        axs = [axs]

    for idx, (model, (folder, use_post)) in enumerate(model_configs.items()):
        if model in ["Low", "GT"]:
            img_path = os.path.join(folder, name)
        else:
            suffix = "_post.png" if use_post else ".png"
            img_path = os.path.join(folder, name_base + suffix)

        if not os.path.exists(img_path):
            axs[idx].set_title(f"{model}\n(缺圖)")
            axs[idx].axis("off")
            continue

        img = Image.open(img_path).convert("RGB")
        if model not in ["Low", "GT"] and use_post:
            img = postprocess_gamma(img)

        axs[idx].imshow(img)
        axs[idx].set_title(model)
        axs[idx].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{name_base}_comparison_vertical.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

print("完成直列圖像比對輸出，儲存於：", output_dir)
