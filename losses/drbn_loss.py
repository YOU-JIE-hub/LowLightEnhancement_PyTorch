import torch
import torch.nn.functional as F


# 建立一維高斯分布，用來生成 SSIM 所需的加權視窗
def gaussian(window_size, sigma):
    x = torch.arange(window_size).float()  # 保證 x 是 Tensor 類型
    gauss = torch.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))  # 高斯分布公式
    return gauss / gauss.sum()  # 正規化，確保總和為 1

# 將 1D 高斯向量擴展為 2D 高斯視窗（高斯核），可套用在影像上做加權平均
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # (window_size, 1)
    _2D_window = _1D_window @ _1D_window.T  # 外積轉成 2D (window_size, window_size)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  # 每個 channel 都套用相同核
    return window

# 核心 SSIM 計算函式（作為 loss 使用），輸入為兩張圖片
def ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()  # 取得通道數（假設為 BCHW 格式）
    window = create_window(window_size, channel).to(img1.device)  # 建立高斯視窗，移到相同裝置（GPU / CPU）

    # 計算平均值 μ1、μ2
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    # μ 的平方與相乘
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # σ（標準差平方）與 協方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # SSIM 常數，用來避免分母為 0（根據論文建議）
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    # SSIM 公式本體
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 轉為 Loss（越大 SSIM 越好 → 我們取 1 - mean）
    return torch.clamp((1 - ssim_map.mean()) / 2, 0, 1)

# 包裝函式，讓 ssim_loss(pred, gt) 更簡潔使用
def ssim_loss(pred, target):
    return ssim(pred, target)

# 色彩一致性損失（Color Constancy Loss）
# 確保 R/G/B 通道亮度平均值接近，避免色偏
def color_constancy_loss(image):
    mean_rgb = image.mean([2, 3])  # 在 H, W 維度上求平均 → (B, C)
    mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]  # 分別取出 R/G/B 通道的平均亮度
    loss = ((mr - mg) ** 2 + (mr - mb) ** 2 + (mg - mb) ** 2).mean()  # 讓三通道亮度差異儘量接近 0
    return loss
