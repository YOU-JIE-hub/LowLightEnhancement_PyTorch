import torch
import torch.nn.functional as F

def single_scale_retinex(img_tensor, sigma):
    # 單尺度 Retinex 增強：對每個通道做高斯模糊，用於去除照明成分
    B, C, H, W = img_tensor.shape
    kernel_size = int(2 * round(3 * sigma) + 1)

    x = torch.arange(kernel_size, dtype=torch.float32, device=img_tensor.device) - kernel_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()

    # 生成 X 與 Y 方向的高斯卷積核（每個通道各自處理）
    kernel_x = gauss.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    kernel_y = gauss.view(1, 1, -1, 1).repeat(C, 1, 1, 1)

    # 分別進行水平與垂直方向的高斯模糊（模擬光照平滑）
    blur = F.conv2d(img_tensor, kernel_x, padding=(0, kernel_size // 2), groups=C)
    blur = F.conv2d(blur, kernel_y, padding=(kernel_size // 2, 0), groups=C)

    return blur

def multi_scale_retinex(img_tensor, sigmas=[15, 80, 250]):
    # 多尺度 Retinex：將多種不同模糊尺度的結果組合
    img_tensor = torch.clamp(img_tensor, min=1e-3)  # 避免 log(0)
    log_img = torch.log(img_tensor)
    msr = torch.zeros_like(img_tensor)

    # 對每一種 sigma 做模糊與 log 差值運算
    for sigma in sigmas:
        blur = single_scale_retinex(img_tensor.clone(), sigma)
        log_blur = torch.log(torch.clamp(blur, min=1e-3))
        msr += log_img - log_blur

    return msr / len(sigmas)  # 平均三個尺度結果

def enhance_retinex(img_tensor):
    # 最終增強：套用多尺度 Retinex 並將輸出正規化至 [0,1]
    msr_tensor = multi_scale_retinex(img_tensor)

    msr_tensor -= msr_tensor.min()
    msr_tensor /= msr_tensor.max()

    return msr_tensor
