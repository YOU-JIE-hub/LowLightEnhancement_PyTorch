import numpy as np
from scipy.ndimage import gaussian_filter

def estimate_illumination_map(image_np):
    # 從輸入影像中取出 R/G/B 三個通道中的最大值，作為每個像素的初始照明值估計
    illumination = np.max(image_np, axis=2)
    return illumination

def refine_illumination_map(illumination, sigma=3):
    # 對初始照明圖進行高斯濾波處理，減少雜訊，保留整體結構
    refined = gaussian_filter(illumination, sigma=sigma)
    return refined

def apply_lime(image_np, refined_illumination, eps=1e-3):
    # 將原始影像每個像素除以照明圖（加上 epsilon 避免除零），達到亮度提升的效果
    illumination_expanded = np.expand_dims(refined_illumination, axis=2)
    enhanced = image_np / (illumination_expanded + eps)
    enhanced = np.clip(enhanced, 0, 1)  # 將結果裁切至合法區間 [0, 1]
    return enhanced

def enhance_lime(image_np):
    """
    主函式：輸入 RGB 影像（可為 [0,255] 或 [0,1]），輸出增強後影像（uint8 格式）
    """
    # 將輸入影像正規化至 [0, 1] 範圍
    if image_np.dtype != np.float32:
        image_np = image_np.astype(np.float32) / 255.0

    # 第一步：估計照明圖
    illum_map = estimate_illumination_map(image_np)

    # 第二步：使用高斯濾波平滑照明圖
    refined_illum = refine_illumination_map(illum_map)

    # 第三步：使用 refined illumination 執行亮度校正
    enhanced = apply_lime(image_np, refined_illum)

    # 將結果轉為 uint8 輸出
    enhanced_uint8 = (enhanced * 255).astype(np.uint8)
    return enhanced_uint8
