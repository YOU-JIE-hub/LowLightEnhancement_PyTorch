import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PairedDataset(Dataset):
    def __init__(self, low_dir, high_dir):
        self.low = sorted([os.path.join(low_dir, f) for f in os.listdir(low_dir) if not f.startswith('.')])
        self.high = sorted([os.path.join(high_dir, f) for f in os.listdir(high_dir) if not f.startswith('.')])
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return min(len(self.low), len(self.high))

    def __getitem__(self, idx):
        return {
            'A': self.transform(Image.open(self.low[idx]).convert('RGB')),
            'B': self.transform(Image.open(self.high[idx]).convert('RGB'))
        }
