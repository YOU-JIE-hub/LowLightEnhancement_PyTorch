import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------- TVLoss：總變分損失，用來抑制增強後圖像的雜訊與細節突變 -------------
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, A):
        batch_size = A.size()[0]          # 批次大小
        h_x = A.size()[2]                 # 高度
        w_x = A.size()[3]                 # 寬度
        count_h = (h_x - 1) * w_x         # 水平方向像素差總數
        count_w = h_x * (w_x - 1)         # 垂直方向像素差總數

        # 計算水平方向變化的平方差總和
        h_tv = torch.pow((A[:, :, 1:, :] - A[:, :, :h_x - 1, :]), 2).sum()
        # 計算垂直方向變化的平方差總和
        w_tv = torch.pow((A[:, :, :, 1:] - A[:, :, :, :w_x - 1]), 2).sum()

        # 回傳總變分損失值（平均）
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


# ----------- SpaLoss：結構一致性損失，用來強化增強圖與原圖的邊緣一致性 ----------
class SpaLoss(nn.Module):
    def __init__(self):
        super(SpaLoss, self).__init__()

        # 四個方向的差分卷積核（用來模擬邊緣強度）
        kernel_left = [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]
        kernel_right = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        kernel_up = [[0, -1, 0], [0, 1, 0], [0, 0, 0]]
        kernel_down = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]

        # 建立卷積權重（固定不訓練）
        self.weight_left = self._create_kernel(kernel_left)
        self.weight_right = self._create_kernel(kernel_right)
        self.weight_up = self._create_kernel(kernel_up)
        self.weight_down = self._create_kernel(kernel_down)

    def _create_kernel(self, kernel):
        # 將 3x3 kernel 轉成 Tensor 並擴展為 RGB 三通道（每個通道都用相同卷積核）
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(3, 1, 1, 1)
        return nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, enhanced, org):
        # 將 RGB 圖像轉為灰階（平均）
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhanced, 1, keepdim=True)

        # 對原圖與增強圖各自計算四個方向的結構變化（邊緣強度）
        D_org = torch.abs(F.conv2d(org_mean, self.weight_left, padding=1) +
                          F.conv2d(org_mean, self.weight_right, padding=1) +
                          F.conv2d(org_mean, self.weight_up, padding=1) +
                          F.conv2d(org_mean, self.weight_down, padding=1))

        D_enhance = torch.abs(F.conv2d(enhance_mean, self.weight_left, padding=1) +
                              F.conv2d(enhance_mean, self.weight_right, padding=1) +
                              F.conv2d(enhance_mean, self.weight_up, padding=1) +
                              F.conv2d(enhance_mean, self.weight_down, padding=1))

        # 最終損失：結構差異越小越好 → 平均絕對差
        loss = torch.mean(torch.abs(D_org - D_enhance))
        return loss


# ----------- ColorConstancyLoss：顏色一致性損失，防止色偏 -------------------
class ColorConstancyLoss(nn.Module):
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, x):
        # 計算每張圖像的 R、G、B 通道平均值
        mean_rgb = torch.mean(x, dim=[2, 3], keepdim=True)
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]

        # 分別計算 RGB 三通道間的均方差，確保顏色分布一致，避免偏色
        drg = torch.pow(mr - mg, 2)
        drb = torch.pow(mr - mb, 2)
        dgb = torch.pow(mb - mg, 2)

        # 返回平均損失
        return (drg + drb + dgb).mean()


# ----------- ExposureLoss：曝光損失，強化整體亮度分布均勻性 ----------------
class ExposureLoss(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6):
        super(ExposureLoss, self).__init__()
        self.mean_val = mean_val  # 目標亮度值（理想平均亮度）
        self.pool = nn.AvgPool2d(patch_size)  # 用區塊平均模擬局部亮度估計

    def forward(self, x):
        # 將 RGB 圖像轉為灰階平均亮度圖
        mean = torch.mean(x, 1, keepdim=True)

        # 區塊平均處理：模擬局部亮度分布
        pool = self.pool(mean)

        # 與目標亮度比較，越接近 0.6 越好
        return torch.mean(torch.pow(pool - self.mean_val, 2))
