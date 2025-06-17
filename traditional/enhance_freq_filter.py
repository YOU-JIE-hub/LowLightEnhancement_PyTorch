import numpy as np
from PIL import Image
import cv2

# === 頻域同態濾波器：針對亮度通道進行增強 ===
def homomorphic_filter(v_channel):
    rows, cols = v_channel.shape
    v_channel = v_channel / 255.0  # 正規化至 [0,1]
    v_channel = np.log1p(v_channel)  # 轉為對數域，模擬光照與反射的乘法關係

    M, N = 2 * rows + 1, 2 * cols + 1  # 頻域尺寸擴展為原圖大小的兩倍
    y, x = np.ogrid[-rows:rows+1, -cols:cols+1]
    D = np.sqrt(x*x + y*y)  # 距離矩陣 D(u,v)

    sigma = 30
    H = 1.5 - 0.5 * np.exp(-(D**2) / (2 * sigma**2))  # 濾波函數 H(u,v)：高頻強化，低頻抑制

    F = np.fft.fft2(v_channel, (M, N))        # 進行傅立葉轉換
    F_shift = np.fft.fftshift(F)              # 中心化
    G_shift = H * F_shift                     # 套用濾波器 H
    G = np.fft.ifftshift(G_shift)             # 還原中心位移
    g = np.fft.ifft2(G)                       # 反傅立葉轉換
    g = np.real(g)                            # 保留實部
    g = g[:rows, :cols]                       # 還原至原圖大小
    g = np.expm1(g)                           # 還原對數
    g = np.clip(g * 255, 0, 255).astype(np.uint8)  # 反正規化並裁切為整數影像

    return g

# === 主函式：對影像套用頻域濾波增強 ===
def enhance_freq_filter(img_path):
    img = Image.open(img_path).convert("RGB")
    ycbcr = img.convert("YCbCr")       # 轉為亮度色度分離空間
    y, cb, cr = ycbcr.split()          # 分離 Y（亮度）、Cb、Cr

    y = y.point(lambda p: min(p * 1.5, 255))  # 預處理：亮度先乘 1.5，避免濾波後過暗

    y_np = np.array(y)                        # 轉為 NumPy 陣列
    filtered_y_np = homomorphic_filter(y_np)  # 濾波增強

    raw_enhanced_img = Image.merge("YCbCr", [  # 合併增強後 Y 與原 Cb/Cr
        Image.fromarray(filtered_y_np), cb, cr
    ]).convert("RGB")                          # 回轉為 RGB 空間

    return raw_enhanced_img
