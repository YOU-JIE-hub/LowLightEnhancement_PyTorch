import torch
import torch.nn as nn
import random

# ImprovedUNet：改良版 U-Net 生成器

class ImprovedUNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, base_ch=64):
        """
        in_c：輸入通道數（一般是3，即RGB）
        out_c：輸出通道數（一般也是3）
        base_ch：基礎通道數（起始卷積數量）
        """
        super().__init__()

        # Encoder（下採樣部分）
        self.e1 = nn.Sequential(
            nn.Conv2d(in_c, base_ch, 4, 2, 1),   # 第1層：卷積，空間降一半
            nn.LeakyReLU(0.2)
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),
            nn.InstanceNorm2d(base_ch*2),        # Instance Normalization
            nn.LeakyReLU(0.2)
        )
        self.e3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1),
            nn.InstanceNorm2d(base_ch*4),
            nn.LeakyReLU(0.2)
        )
        self.e4 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1),
            nn.InstanceNorm2d(base_ch*8),
            nn.LeakyReLU(0.2)
        )

        # Decoder（上採樣部分）
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*8, base_ch*4, 4, 2, 1),  # 上採樣回一半
            nn.InstanceNorm2d(base_ch*4),
            nn.ReLU()
        )
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4*2, base_ch*2, 4, 2, 1),  # 跟encoder skip連接，通道數要*2
            nn.InstanceNorm2d(base_ch*2),
            nn.ReLU()
        )
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2*2, base_ch, 4, 2, 1),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU()
        )

        # 最後一層：直接輸出目標圖像（不用norm）
        self.d4 = nn.ConvTranspose2d(base_ch*2, out_c, 4, 2, 1)

    def forward(self, x):
        """
        前向傳播：
        - 編碼一路保存下來
        - 解碼一路接上跳連結
        """
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        d1 = self.d1(x4)
        d2 = self.d2(torch.cat([d1, x3], 1))  # 跟 x3 skip connect
        d3 = self.d3(torch.cat([d2, x2], 1))  # 跟 x2 skip connect
        out = self.d4(torch.cat([d3, x1], 1)) # 跟 x1 skip connect

        # 最後輸出通過 tanh 映射到 (0,1)
        return (torch.tanh(out) + 1) / 2

# Discriminator：PatchGAN 判別器

class Discriminator(nn.Module):
    def __init__(self, in_c=3, base_ch=64):
        """
        in_c：輸入通道數
        base_ch：起始卷積通道數
        """
        super().__init__()
        self.net = nn.Sequential(
            # 第一層，不加norm
            nn.utils.spectral_norm(nn.Conv2d(in_c, base_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            # 第二層
            nn.utils.spectral_norm(nn.Conv2d(base_ch, base_ch*2, 4, 2, 1)),
            nn.InstanceNorm2d(base_ch*2),
            nn.LeakyReLU(0.2),
            # 第三層
            nn.utils.spectral_norm(nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1)),
            nn.InstanceNorm2d(base_ch*4),
            nn.LeakyReLU(0.2),
            # 最後輸出層（通道數 1）
            nn.Conv2d(base_ch*4, 1, 4, 1, 1)
        )

    def forward(self, x):
        """
        判別器前向傳播
        """
        return self.net(x)

# ReplayBuffer：歷史緩存池

class ReplayBuffer():
    def __init__(self, max_size=50):
        """
        max_size：緩存最多保存 50 張舊圖
        """
        self.data, self.max_size = [], max_size

    def push_and_pop(self, images):
        """
        保存並選擇性回取過去的舊圖（防止判別器過快收斂）
        - 有 50% 機率從 buffer 裡拿舊圖
        - 有 50% 機率直接用新生成的圖
        """
        out = []
        for img in images:
            img = img.unsqueeze(0)  # 增加 batch 維度
            if len(self.data) < self.max_size:
                # Buffer 未滿時直接存入
                self.data.append(img)
                out.append(img)
            else:
                if random.random() > 0.5:
                    # 取出一張舊圖，用新圖取代
                    idx = random.randint(0, self.max_size-1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = img
                    out.append(tmp)
                else:
                    # 不從 buffer，直接用新的
                    out.append(img)
        return torch.cat(out, dim=0)  # 合併成 batch
