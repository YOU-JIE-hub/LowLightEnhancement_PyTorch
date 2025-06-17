import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LOLPairedDataset(Dataset):
    def __init__(self, low_dir, high_dir):
        self.low_images = sorted(glob(os.path.join(low_dir, '*')))
        self.high_images = sorted(glob(os.path.join(high_dir, '*')))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, index):
        low_image = Image.open(self.low_images[index]).convert('RGB')
        if self.transform:
            low_image = self.transform(low_image)
        return low_image, 0  # 只回傳 low image tensor
