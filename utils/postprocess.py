from PIL import Image
import numpy as np

def postprocess_gamma(image: Image.Image, gamma: float = 0.8, gain: float = 1.1) -> Image.Image:
    """
    對 PIL 圖片進行 gamma + gain 調整，取消亮度判斷，直接套用保守後處理。
    Args:
        image: 輸入 PIL 圖片。
        gamma: gamma 校正值（>1 暗化，<1 提亮）。
        gain: 增益值，乘上整張圖。
    Returns:
        處理後的 PIL 圖片。
    """
    img_np = np.asarray(image).astype(np.float32) / 255.0
    img_np = gain * np.power(img_np, gamma)
    img_np = np.clip(img_np, 0, 1)
    return Image.fromarray((img_np * 255).astype(np.uint8))
