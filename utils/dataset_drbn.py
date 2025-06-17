import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DRBNDataset(Dataset):
    def __init__(self, low_dir, high_dir):
        self.low_list = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir)])
        self.high_list = sorted([os.path.join(high_dir, f) for f in os.listdir(high_dir)])
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.low_list)

    def __getitem__(self, idx):
        low_img = self.transform(Image.open(self.low_list[idx]).convert('RGB'))
        high_img = self.transform(Image.open(self.high_list[idx]).convert('RGB'))
        return low_img, high_img
