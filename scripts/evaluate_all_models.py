import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import piq
import torchvision.transforms.functional as TF

gt_dir = r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\data\Raw\high_val"
save_dir = r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results\Comparison"
os.makedirs(save_dir, exist_ok=True)

# 用 _val 資料夾
model_configs = {
    "FreqFilter":         ("FreqFilter_val", "{name}.png"),
    "LIME":               ("LIME_val", "{name}.png"),
    "RetinexNet":         ("RetinexNet_val", "{name}.png"),
    "RetinexTraditional": ("RetinexTraditional_val", "{name}.png"),
    "ZeroDCE":            ("ZeroDCE_val", "{name}.png"),
    "EnlightenGAN":       ("EnlightenGAN_val", "{name}.png"),
    "DRBN":               ("DRBN_val", "{name}.png")
}

def compute_fullref(gt_img, pred_img):
    gt_np = np.array(gt_img)
    pred_np = np.array(pred_img.resize(gt_img.size))
    if gt_np.shape[0] < 7 or gt_np.shape[1] < 7:
        raise ValueError("圖太小")
    return psnr(gt_np, pred_np, data_range=255), ssim(gt_np, pred_np, channel_axis=-1, data_range=255)

def compute_noref(tensor):
    tensor = tensor.unsqueeze(0).clamp(0, 1)
    brisque_score = piq.brisque(tensor).item()
    pi_score = 0.5 * (10 + brisque_score)
    return brisque_score, pi_score

def plot_radar_chart(model_list, filename):
    metrics = ["PSNR", "SSIM", "BRISQUE", "PI"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for model in model_list:
        if model not in results: continue
        values = [results[model][m] for m in metrics]
        norm_vals = values.copy()
        norm_vals[2] = 100 - norm_vals[2]  # BRISQUE 反向
        norm_vals = (np.array(norm_vals) - min_vals) / (max_vals - min_vals + 1e-6)
        norm_vals = norm_vals.tolist()
        norm_vals += [norm_vals[0]]
        ax.plot(angles, norm_vals, label=model)
        ax.fill(angles, norm_vals, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close()

gt_list = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
print(f"\n📝 共 {len(gt_list)} 張 Val Ground Truth 圖片，開始模型評估...\n")

results = {}

for model, (folder, pattern) in model_configs.items():
    psnr_scores, ssim_scores = [], []
    brisque_scores, pi_scores = [], []
    count = 0

    for name in tqdm(gt_list, desc=f"評估 {model:>18s}"):
        name_base = os.path.splitext(name)[0]
        gt_path = os.path.join(gt_dir, name)

        try:
            gt_img = Image.open(gt_path).convert("RGB")
            if gt_img.width < 7 or gt_img.height < 7:
                continue
        except:
            continue

        pred_path = os.path.join(
            r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\results",
            folder, pattern.format(name=name_base)
        )
        if not os.path.exists(pred_path): continue

        try:
            pred_img = Image.open(pred_path).convert("RGB")
            if pred_img.width < 7 or pred_img.height < 7:
                continue
            p, s = compute_fullref(gt_img, pred_img)
            t = TF.to_tensor(pred_img)
            bris, pi_val = compute_noref(t)
            psnr_scores.append(p)
            ssim_scores.append(s)
            brisque_scores.append(bris)
            pi_scores.append(pi_val)
            count += 1
        except:
            continue

    if count > 0:
        results[model] = {
            "PSNR": np.mean(psnr_scores),
            "SSIM": np.mean(ssim_scores),
            "BRISQUE": np.mean(brisque_scores),
            "PI": np.mean(pi_scores),
            "Count": count
        }

print("\n📊 模型畫質評估總結：")
print("{:<20} {:>8} {:>8} {:>10} {:>8} {:>10}".format("模型", "PSNR", "SSIM", "BRISQUE", "PI", "張數"))
for model, score in results.items():
    print(f"{model:<20} {score['PSNR']:>8.2f} {score['SSIM']:>8.3f} {score['BRISQUE']:>10.2f} {score['PI']:>8.2f} {score['Count']:>10}")

df = pd.DataFrame.from_dict(results, orient="index")
df.index.name = "Model"
df.reset_index(inplace=True)
csv_path = os.path.join(save_dir, "quality_val.csv")
df.to_csv(csv_path, index=False)

metric_cols = ["PSNR", "SSIM", "BRISQUE", "PI"]
metric_data = df[metric_cols].astype(float)
max_vals = metric_data.max()
min_vals = metric_data.min()

plot_radar_chart(df["Model"].tolist(), "quality_val_radar.png")
plot_radar_chart(["RetinexNet", "ZeroDCE", "DRBN", "EnlightenGAN"], "radar_val_group1_dl.png")
plot_radar_chart(["RetinexTraditional", "FreqFilter", "LIME"], "radar_val_group2_traditional.png")

print(f"\n指標總表已儲存：{csv_path}")
print("雷達圖已儲存：quality_val_radar.png, radar_val_group1_dl.png, radar_val_group2_traditional.png")
