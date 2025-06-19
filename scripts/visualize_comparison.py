import os 
import random
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Microsoft JhengHei'
from utils.postprocess import postprocess_gamma

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
low_dir = os.path.join(project_root, "data", "Raw", "low_val")
gt_dir = os.path.join(project_root, "data", "Raw", "high_val")
output_dir = os.path.join(project_root, "results", "Comparison")
os.makedirs(output_dir, exist_ok=True)

model_order = [
    "Low", "GT", "FreqFilter",
    "RetinexNet", "ZeroDCE", "RetinexTraditional",
    "LIME", "EnlightenGAN", "DRBN"
]

model_configs = {
    "Low":               (low_dir, False),
    "GT":                (gt_dir, False),
    "FreqFilter":        (os.path.join(project_root, "results", "FreqFilter", "post"), True),
    "RetinexNet":        (os.path.join(project_root, "results", "RetinexNet"), False),
    "ZeroDCE":           (os.path.join(project_root, "results", "ZeroDCE", "post"), True),
    "RetinexTraditional":(os.path.join(project_root, "results", "Retinex_Traditional"), False),
    "LIME":              (os.path.join(project_root, "results", "LIME"), False),
    "EnlightenGAN":      (os.path.join(project_root, "results", "EnlightenGAN"), False),
    "DRBN":              (os.path.join(project_root, "results", "DRBN"), False),
}

sample_names = random.sample(
    [f for f in os.listdir(gt_dir) if f.endswith(".png")], 5
)

for name in sample_names:
    name_base = os.path.splitext(name)[0]
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))  # 3x3

    for idx, model in enumerate(model_order):
        row, col = divmod(idx, 3)
        folder, use_post = model_configs[model]

        if model in ["Low", "GT"]:
            img_path = os.path.join(folder, name)
        else:
            suffix = "_post.png" if use_post else ".png"
            img_path = os.path.join(folder, name_base + suffix)

        if not os.path.exists(img_path):
            axs[row][col].set_title(f"{model}\n(缺圖)")
            axs[row][col].axis("off")
            continue

        img = Image.open(img_path).convert("RGB")
        if model not in ["Low", "GT"] and use_post:
            img = postprocess_gamma(img)

        axs[row][col].imshow(img)
        axs[row][col].set_title(model)
        axs[row][col].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{name_base}_comparison_grid.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

print("完成增強輸出比對圖，儲存於：", output_dir)
