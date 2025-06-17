import os
from PIL import Image
from torch.utils.data import Dataset

class LOLPairedDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None):
        self.low_images = sorted(os.listdir(low_dir))
        self.high_images = sorted(os.listdir(high_dir))
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform  #直接使用外部傳入的 transform

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.low_images[idx])
        high_path = os.path.join(self.high_dir, self.high_images[idx])

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img
