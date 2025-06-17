import torch
import torch.nn as nn

# 🔹 基本卷積層（含 BatchNorm 與 ReLU）
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),  # 卷積
            nn.BatchNorm2d(out_channels),  # 正規化
            nn.ReLU(inplace=True)  # 非線性激活
        )

    def forward(self, x):
        return self.conv(x)  # 回傳經卷積+BN+ReLU處理的結果

# 🔹 遞迴殘差區塊（Recursive Block）
class RecursiveBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels)
        self.conv2 = ConvLayer(channels, channels)

    def forward(self, x):
        residual = x  # 保留輸入作為殘差
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual  # 加上殘差以穩定訓練、提升表現

# 🔹 Band 模組（可疊多層 RecursiveBlock）
class BandBlock(nn.Module):
    def __init__(self, channels, recursions):
        super().__init__()
        self.recur = nn.Sequential(
            *[RecursiveBlock(channels) for _ in range(recursions)]  # 疊多層遞迴模塊
        )

    def forward(self, x):
        return self.recur(x)

# 🔹 DRBN 主體模型
class DRBN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64, recursions=3):
        super().__init__()
        self.head = ConvLayer(in_channels, features)  # 初始卷積層
        self.band1 = BandBlock(features, recursions)  # 第一個 Band（多層遞迴）
        self.band2 = BandBlock(features, recursions)  # 第二個 Band
        self.tail = nn.Conv2d(features, out_channels, 3, 1, 1)  # 輸出層

    def forward(self, x):
        out = self.head(x)      # 初始特徵提取
        out = self.band1(out)   # 通過 Band1
        out = self.band2(out)   # 通過 Band2
        out = self.tail(out)    # 預測輸出圖像
        out = torch.tanh(out)   # 將結果限制在 [-1, 1]
        return torch.clamp(out, 0, 1)  # 最終限制在 [0, 1] 範圍（符合圖像格式）
