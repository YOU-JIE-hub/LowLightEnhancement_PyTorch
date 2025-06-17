import torch
import torch.nn as nn

class ZeroDCE(nn.Module):
    def __init__(self):
        super(ZeroDCE, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        # 建立連續 7 層卷積，前 6 層為 32 channel，維持特徵維度
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32, 32, 3, 1, 1)

        # 第 7 層輸出為 24 channel，代表 8 組 RGB 增強曲線參數（8 × 3 = 24）
        self.conv7 = nn.Conv2d(32, 24, 3, 1, 1)

    def forward(self, x):
        x_in = x  # 保留原始輸入，用於後續運算

        # 通過 ReLU 激活的多層卷積提取特徵
        x = self.relu(self.conv1(x_in))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))

        # 最後輸出使用 tanh 限制範圍在 [-1, 1]，代表亮度增強的曲線參數 A
        x = torch.tanh(self.conv7(x))

        return x  # 回傳 A 值，用於曲線映射增強（在推論程式處理）
