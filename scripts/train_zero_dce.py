import os 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from networks.zero_dce import ZeroDCE
from utils.dataset_lol_dce import LOLPairedDataset
from utils.dce_utils import apply_curve
from losses.zero_dce_loss import TVLoss, ColorConstancyLoss, ExposureLoss

def train():
    batch_size = 8
    num_epochs = 100
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lowlight_dir = os.path.join(project_root, "data", "Raw", "low")
    highlight_dir = os.path.join(project_root, "data", "Raw", "high")
    save_dir = os.path.join(project_root, "checkpoints", "ZeroDCE")
    preview_dir = os.path.join(project_root, "results", "ZeroDCE", "preview")
    os.makedirs(lowlight_dir, exist_ok=True)
    os.makedirs(highlight_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    train_dataset = LOLPairedDataset(lowlight_dir, highlight_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ZeroDCE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_tv = TVLoss()
    loss_color = ColorConstancyLoss()
    loss_exposure = ExposureLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"ZeroDCE Epoch {epoch}/{num_epochs}", ncols=100)

        for low_img, _ in loop:
            low_img = low_img.to(device)
            optimizer.zero_grad()

            A = model(low_img)  # 預測曲線參數 A
            enhanced = apply_curve(low_img, A)  # 應用曲線增強影像

            loss = loss_color(enhanced) + loss_exposure(enhanced) + loss_tv(A)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch:03d}/{num_epochs}] Total Loss: {avg_loss:.4f}")

        if epoch % 20 == 0:
            ckpt_path = os.path.join(save_dir, f"zero_dce_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint 儲存於: {ckpt_path}")
            model.train()
    print("ZeroDCE 訓練完成")

if __name__ == "__main__":
    train()
