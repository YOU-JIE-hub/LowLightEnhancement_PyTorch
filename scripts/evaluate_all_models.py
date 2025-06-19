import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms.functional as TF
import piq

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gt_dir = os.path.join(project_root, "data", "Raw", "high_val")
model_base = os.path.join(project_root, "results")

model_configs = {
    "FreqFilter":         ("FreqFilter", "{name}.png"),
    "LIME":               ("LIME", "{name}.png"),
    "RetinexNet":         ("RetinexNet", "{name}.png"),
    "Retinex_Traditional": ("Retinex_Traditional", "{name}.png"),
    "ZeroDCE":            ("ZeroDCE", "{name}.png"),
    "EnlightenGAN":       ("EnlightenGAN", "{name}.png"),
    "DRBN":               ("DRBN", "{name}.png")
}

def compute_metrics(gt_img, pred_img):
    gt_np = np.array(gt_img)
    pred_np = np.array(pred_img.resize(gt_img.size))
    if gt_np.shape[0] < 7 or gt_np.shape[1] < 7:
        raise ValueError("圖像太小無法計算")
    psnr_val = psnr(gt_np, pred_np, data_range=255)
    ssim_val = ssim(gt_np, pred_np, channel_axis=-1, data_range=255)
    pred_tensor = TF.to_tensor(pred_img).clamp(0, 1).unsqueeze(0)
    brisque_val = piq.brisque(pred_tensor).item()
    pi_score = 0.5 * (10 + brisque_val)
    return psnr_val, ssim_val, brisque_val, pi_score

def main():
    if not os.path.exists(gt_dir):
        rel_hint = os.path.join("data", "Raw", "high_val")
        print(f"找不到 high_val 資料夾：{gt_dir}\n請先把圖片放入 {rel_hint} 再執行本程式。")
        return

    gt_list = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if len(gt_list) == 0:
        print(f"high_val 資料夾 {gt_dir} 中沒有圖片。請確認有放置驗證圖片。")
        return

    print(f"\n共 {len(gt_list)} 張驗證圖片，開始評估各模型輸出品質...\n")

    results = {}

    for model, (folder, pattern) in model_configs.items():
        model_dir = os.path.join(model_base, folder)
        if not os.path.exists(model_dir):
            print(f"找不到 {model} 的結果資料夾\n請先執行模型推論，如：python -m scripts.infer_{model.lower()}")
            continue

        psnr_scores, ssim_scores = [], []
        brisque_scores, pi_scores = [], []
        count = 0
        missing = 0

        for name in tqdm(gt_list, desc=f"評估 {model:>18s}"):
            name_base = os.path.splitext(name)[0]
            gt_path = os.path.join(gt_dir, name)
            pred_path = os.path.join(model_dir, pattern.format(name=name_base))

            if not os.path.exists(pred_path):
                missing += 1
                continue

            try:
                gt_img = Image.open(gt_path).convert("RGB")
                pred_img = Image.open(pred_path).convert("RGB")
                p, s, b, pi_val = compute_metrics(gt_img, pred_img)
                psnr_scores.append(p)
                ssim_scores.append(s)
                brisque_scores.append(b)
                pi_scores.append(pi_val)
                count += 1
            except Exception as e:
                print(f"讀取圖片失敗：{pred_path}，錯誤：{e}")
                continue

        if count > 0:
            results[model] = {
                "PSNR": np.mean(psnr_scores),
                "SSIM": np.mean(ssim_scores),
                "BRISQUE": np.mean(brisque_scores),
                "PI": np.mean(pi_scores),
                "Count": count,
                "Missing": missing
            }

    print("\n模型畫質評估：")
    print("{:<26} {:>8} {:>8} {:>10} {:>8} {:>6}".format("模型", "PSNR", "SSIM", "BRISQUE", "PI", "張數"))
    for model, score in results.items():
        print("{:<28} {:>8.2f} {:>8.3f} {:>10.2f} {:>8.2f} {:>6}".format(
            model, score["PSNR"], score["SSIM"], score["BRISQUE"], score["PI"], score["Count"]
        ))

if __name__ == "__main__":
    main()
