from PIL import Image
from torchvision import transforms
import os

def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (W, H)
    img_resized = img.resize(target_size, Image.BICUBIC)
    to_tensor = transforms.ToTensor()
    tensor_img = to_tensor(img_resized)
    return tensor_img, orig_size

def save_image(tensor, save_path, orig_size):
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.cpu().clamp(0, 1))
    image = image.resize(orig_size, Image.BICUBIC)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
