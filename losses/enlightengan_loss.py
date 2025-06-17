import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math


def d_hinge_loss(real_pred, fake_pred):
    """
    判別器 Hinge loss：
    - 期望真實圖 real_pred ≥ 1
    - 期望假圖 fake_pred ≤ -1
    - ReLU(1 - real_pred) + ReLU(1 + fake_pred)，取平均
    """
    return 0.5 * (torch.mean(F.relu(1. - real_pred)) + torch.mean(F.relu(1. + fake_pred)))

def g_hinge_loss(fake_pred):
    """
    生成器 Hinge loss：
    - 期望生成圖 fake_pred 越大越好（最大化）
    """
    return -torch.mean(fake_pred)

def color_loss(x):
    """
    顏色一致性損失 (Color Constancy Loss)：
    - 強制 R、G、B 通道的差異減少
    - (r-g)^2 + (r-b)^2 + (g-b)^2 平均
    """
    r, g, b = x[:, 0], x[:, 1], x[:, 2]  # 拆分 RGB 三個通道
    return torch.mean((r-g)**2 + (r-b)**2 + (g-b)**2)

def laplacian(x):
    """
    計算圖像的 Laplacian（邊緣增強）：
    - 使用固定卷積核 [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    - 對每個通道分組卷積
    """
    k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    k = k.repeat(x.shape[1], 1, 1, 1)  # 複製成與通道數相符
    return F.conv2d(x, k, padding=1, groups=x.shape[1])

def lap_loss(pred, gt):
    """
    Laplacian loss：
    - 強化邊緣資訊
    - 比較 pred 和 gt 的 Laplacian 特徵差異（L1 Loss）
    """
    return F.l1_loss(laplacian(pred), laplacian(gt))

def tv_loss(x):
    """
    總變分損失（Total Variation Loss）：
    - 平滑圖像，防止輸出過多噪點
    - 計算水平方向與垂直方向的像素差
    """
    return (torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).mean() + torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).mean())

def ssim_loss(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    SSIM (Structural Similarity) 損失：
    - 衡量兩張圖像在結構上的相似度
    - 較小代表越不相似
    """
    def gaussian(window_size, sigma):
        # 生成一維高斯分布 (不標準化)
        gauss = torch.tensor([math.exp(-(x - window_size//2)**2 / (2*sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()  # 正規化成總和為 1

    _1d = gaussian(window_size, 1.5).unsqueeze(1)  # 轉成列向量
    window = _1d @ _1d.t()  # 外積生成二維高斯核
    window = window.expand(img1.shape[1], 1, window_size, window_size).to(img1.device)

    # 均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2

    # 方差與協方差
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2

    # SSIM 計算公式
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    # 損失為 (1 - SSIM) / 2，範圍在 [0,1]
    return torch.clamp((1 - ssim_map.mean())/2, 0, 1)


class VGGPerceptual(nn.Module):
    def __init__(self):
        """
        VGG 感知損失：
        - 使用 VGG19 前 16 層特徵（即到 relu4_1）
        - 固定權重 (no gradient)
        - 用來比較生成圖與真實圖的深層特徵差異
        """
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:16].eval()  # 載入預訓練 VGG19 的前 16 層
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet RGB 平均
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)    # ImageNet RGB 標準差

    def forward(self, x, y):
        """
        x: 預測圖
        y: 真實圖
        流程：
        - 先做 ImageNet 標準化
        - 經過 VGG 提取特徵
        - 用 L1 損失比較特徵差異
        """
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        fx = (x - mean) / std  # 標準化預測圖
        fy = (y - mean) / std  # 標準化真實圖
        return F.l1_loss(self.vgg(fx), self.vgg(fy))
