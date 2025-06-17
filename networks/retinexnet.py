import torch
import torch.nn as nn
import torch.nn.functional as F

# 功能：將輸入圖像分解成反射圖 (R) 和照明圖 (I)
class UNetDecomNet(nn.Module):
    def __init__(self):
        super(UNetDecomNet, self).__init__()
        # Encoder：三層卷積 + ReLU，逐漸增加通道數
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(inplace=True))   # 輸入3通道，輸出32通道
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True))  # 輸出64通道
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True)) # 輸出128通道
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True))# 輸出256通道
        # Decoder：上取樣後，與Encoder層concat，再卷積
        self.upconv3 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, 1, 1), nn.ReLU(inplace=True))
        self.upconv2 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, 1, 1), nn.ReLU(inplace=True))
        self.upconv1 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, 1, 1), nn.ReLU(inplace=True))
        # 輸出層：預測反射圖R（3通道）與照明圖I（1通道）
        self.out_R = nn.Conv2d(32, 3, 3, 1, 1)
        self.out_I = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, x):
        # Encoder路徑
        x1 = self.conv1(x)                  # 第一層卷積
        x2 = self.conv2(F.max_pool2d(x1, 2)) # 第二層：池化後卷積
        x3 = self.conv3(F.max_pool2d(x2, 2)) # 第三層
        x4 = self.conv4(F.max_pool2d(x3, 2)) # 第四層

        # Decoder路徑
        up3 = F.interpolate(x4, scale_factor=2, mode='nearest')  # 上取樣
        up3 = self.upconv3(torch.cat([up3, x3], dim=1))          # 與編碼器對應層concat後卷積

        up2 = F.interpolate(up3, scale_factor=2, mode='nearest')
        up2 = self.upconv2(torch.cat([up2, x2], dim=1))

        up1 = F.interpolate(up2, scale_factor=2, mode='nearest')
        up1 = self.upconv1(torch.cat([up1, x1], dim=1))

        # 輸出層，經Sigmoid壓縮到[0,1]範圍
        R = torch.sigmoid(self.out_R(up1))   # 反射圖
        I = torch.sigmoid(self.out_I(up1))   # 照明圖

        return R, I

# 功能：以反射圖R和初步照明圖I為輸入，進一步增強光照與色彩
class EnhanceNetDeep(nn.Module):
    def __init__(self):
        super(EnhanceNetDeep, self).__init__()
        # Encoder部分
        self.enc1 = nn.Sequential(nn.Conv2d(4, 64, 3, 1, 1), nn.ReLU(True))    # 輸入 R(3) + I(1) = 4通道
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True))  # 下採樣（stride=2）
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(True))

        # 中間Bottleneck部分
        self.middle = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True)
        )

        # Decoder部分，帶跳接（skip connections）
        self.dec3 = nn.Sequential(nn.Conv2d(512 + 256, 256, 3, 1, 1), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, 1, 1), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, 1, 1), nn.ReLU(True))

        # 輸出層
        self.out_conv_I = nn.Conv2d(64, 1, 3, 1, 1)   # 輸出增強後的亮度圖（I_hat）
        self.out_conv_color = nn.Conv2d(64, 3, 3, 1, 1) # 輸出色彩修正圖（color map）

    def forward(self, R, I):
        # 輸入組合：將反射圖R與照明圖I串接
        x = torch.cat([R, I], dim=1)

        # Encoder路徑
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        m = self.middle(e4)

        # Decoder路徑
        d3 = self.dec3(torch.cat([F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=False), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False), e1], 1))

        I_out = torch.sigmoid(self.out_conv_I(d1))    # 經Sigmoid壓縮到[0,1]，輸出增強亮度圖
        color_map = torch.tanh(self.out_conv_color(d1))# 經Tanh壓縮到[-1,1]，輸出色彩校正圖

        return I_out, color_map
