import torch
import torch.nn.functional as F

# 目的是讓照明圖 I 更加平滑，並且遵循反射圖 R 的邊緣結構
def illumination_smoothness_loss(I, R):
    def gradient(img, direction):
        # 計算影像的梯度
        if direction == 'x':
            # x 方向梯度：沿著寬度方向相減
            return img[:, :, :, 1:] - img[:, :, :, :-1]
        else:
            # y 方向梯度：沿著高度方向相減
            return img[:, :, 1:, :] - img[:, :, :-1, :]

    # 將反射圖 R 轉為灰階，避免彩色分量干擾
    R_gray = 0.299 * R[:, 0:1] + 0.587 * R[:, 1:2] + 0.114 * R[:, 2:3]

    # 計算 I 的 x 方向平滑損失
    loss = (gradient(I, 'x') * torch.exp(-10 * F.avg_pool2d(torch.abs(gradient(R_gray, 'x')), 3, 1, 1))).mean()
    # 計算 I 的 y 方向平滑損失
    loss += (gradient(I, 'y') * torch.exp(-10 * F.avg_pool2d(torch.abs(gradient(R_gray, 'y')), 3, 1, 1))).mean()
    # 回傳總損失
    return loss

# 目的是讓反射圖 R 的邊緣結構與原始圖 img 的邊緣保持一致
def structure_loss(R, img):
    def gradient(img, direction):
        # 計算影像的梯度
        if direction == 'x':
            return img[:, :, :, 1:] - img[:, :, :, :-1]
        else:
            return img[:, :, 1:, :] - img[:, :, :-1, :]

    # L1 損失：比較反射圖 R 與原圖 img 在 x 和 y 方向梯度上的差異
    return F.l1_loss(gradient(R, 'x'), gradient(img, 'x')) + F.l1_loss(gradient(R, 'y'), gradient(img, 'y'))

# 目的是讓增強後圖像 pred 與目標圖像 target 在整體色調上接近
def color_loss(pred, target):
    # 取 pred 和 target 的整體平均色（沿著高度、寬度取平均），再算 L1 loss
    return F.l1_loss(pred.mean((2, 3), keepdim=True), target.mean((2, 3), keepdim=True))

# 目的是讓增強後圖像的平均亮度落在合理範圍內
def brightness_reg_loss(pred, lower=0.4, upper=0.7):
    mean_brightness = pred.mean()  # 計算增強圖像的平均亮度
    loss_low = F.relu(lower - mean_brightness)  # 若亮度低於下限，產生懲罰
    loss_high = F.relu(mean_brightness - upper)  # 若亮度高於上限，產生懲罰
    return loss_low + loss_high  # 合併上下界的懲罰作為總損失

# 目的是讓 I 的整體亮度均值接近目標值 target_mean（預設為 0.5）
def illumination_mean_loss(I, target_mean=0.5):
    mean_I = I.mean()  # 計算 I 的均值
    return F.mse_loss(mean_I, torch.tensor(target_mean, device=I.device))  # 與目標均值計算 MSE

# 目的是加強圖像的邊緣細節
def laplacian_loss(img):
    B, C, H, W = img.shape  # 取得批次大小、通道數、高度、寬度
    # 定義 Laplacian 核（邊緣偵測用）
    lap_kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32, device=img.device)
    # 擴展 Laplacian 核，使每個通道分開卷積（groups=C）
    lap_kernel = lap_kernel.expand(C, 1, 3, 3)
    # 對每個通道進行 2D 卷積（padding=1 保持尺寸）
    lap = F.conv2d(img, lap_kernel, padding=1, groups=C)
    # 回傳 Laplacian 絕對值的均值作為損失
    return torch.mean(torch.abs(lap))
