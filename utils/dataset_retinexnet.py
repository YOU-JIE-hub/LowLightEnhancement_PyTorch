import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LOLPatchDataset(Dataset):
    def __init__(self, low_dir, high_dir, patch_size=128):
        self.low_paths = sorted(os.listdir(low_dir))
        self.high_paths = sorted(os.listdir(high_dir))
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.patch_size = patch_size
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low = Image.open(os.path.join(self.low_dir, self.low_paths[idx])).convert('RGB')
        high = Image.open(os.path.join(self.high_dir, self.high_paths[idx])).convert('RGB')
        low = self.transform(low)
        high = self.transform(high)
        _, H, W = low.shape
        ps = self.patch_size
        rnd_h = random.randint(0, H - ps)
        rnd_w = random.randint(0, W - ps)
        return low[:, rnd_h:rnd_h+ps, rnd_w:rnd_w+ps], high[:, rnd_h:rnd_h+ps, rnd_w:rnd_w+ps]
