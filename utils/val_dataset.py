import os
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset

# UnpairedEnlightenDataset - 訓練集資料類別
# - 讀取 low-light 與 high-light 影像
# - 資料增強（隨機左右翻轉、Color Jitter）
# - gamma 校正

class UnpairedEnlightenDataset(Dataset):
    def __init__(self, low_dir, high_dir, size=(256,256), gamma=2.2):
        """
        low_dir：低光影像資料夾
        high_dir：正常光影像資料夾
        size：resize 到固定尺寸 (預設 256x256)
        gamma：gamma 矯正參數（預設 2.2）
        """
        # 讀取並排序 low-light 與 high-light 檔案名稱
        self.low_files = sorted([f for f in os.listdir(low_dir) if f.lower().endswith(('jpg','png'))])
        self.high_files = sorted([f for f in os.listdir(high_dir) if f.lower().endswith(('jpg','png'))])

        # 定義資料轉換操作
        self.resize = transforms.Resize(size)           # 調整尺寸
        self.jitter = transforms.ColorJitter(0.2,0.2,0.2) # 色彩抖動（這個版本其實沒用到）
        self.to_tensor = transforms.ToTensor()          # 轉成 tensor 格式 [0,1]

        self.low_dir, self.high_dir = low_dir, high_dir  # 保存資料夾路徑
        self.inv_gamma = 1.0 / gamma                     # 預先計算 gamma 倒數

    def __len__(self):
        # 資料集長度為 low 與 high 中較大的（可支援不等長）
        return max(len(self.low_files), len(self.high_files))

    def __getitem__(self, idx):
        # 根據索引讀取 low-light 與 high-light 圖片
        low  = Image.open(os.path.join(self.low_dir, self.low_files[idx % len(self.low_files)])).convert('RGB')
        high = Image.open(os.path.join(self.high_dir, self.high_files[idx % len(self.high_files)])).convert('RGB')

        # Resize到固定大小
        low, high = self.resize(low), self.resize(high)

        # 有50%機率左右翻轉（資料增強）
        if random.random() < 0.5:
            low, high = TF.hflip(low), TF.hflip(high)

        # 轉成 tensor，同時對 low-light 做 gamma 校正
        low = self.to_tensor(low).pow(self.inv_gamma)
        high = self.to_tensor(high)

        return low, high

# ValDataset - 驗證集資料類別
# - 讀取 low-light 與 high-light
# - 只有 resize 與 gamma 校正

class ValDataset(Dataset):
    def __init__(self, low_dir, high_dir, size=(256,256), gamma=2.2):
        """
        low_dir：低光影像資料夾
        high_dir：正常光影像資料夾
        size：resize 到固定尺寸 (預設 256x256)
        gamma：gamma 矯正參數（預設 2.2）
        """
        # 同樣讀取並排序檔案名稱
        self.low_files = sorted([f for f in os.listdir(low_dir) if f.lower().endswith(('jpg','png'))])
        self.high_files = sorted([f for f in os.listdir(high_dir) if f.lower().endswith(('jpg','png'))])

        self.resize = transforms.Resize(size)          # 調整尺寸
        self.to_tensor = transforms.ToTensor()          # 轉 tensor
        self.low_dir, self.high_dir = low_dir, high_dir  # 保存資料夾路徑
        self.inv_gamma = 1.0 / gamma                     # gamma倒數

    def __len__(self):
        # 驗證集取 low 與 high 中「較小的長度」
        return min(len(self.low_files), len(self.high_files))

    def __getitem__(self, idx):
        # 直接按順序取對應圖片，不做資料增強
        low  = Image.open(os.path.join(self.low_dir, self.low_files[idx])).convert('RGB')
        high = Image.open(os.path.join(self.high_dir, self.high_files[idx])).convert('RGB')

        # Resize
        low, high = self.resize(low), self.resize(high)

        # gamma 校正
        low = self.to_tensor(low).pow(self.inv_gamma)
        high = self.to_tensor(high)

        return low, high
