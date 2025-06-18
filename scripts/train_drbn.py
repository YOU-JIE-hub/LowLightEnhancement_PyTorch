import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from networks.drbn import DRBN
from utils.dataset_drbn import DRBNDataset
from losses.drbn_loss import ssim_loss, color_constancy_loss
from lpips import LPIPS


def train():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(project_root, "checkpoints", "DRBN")
    preview_dir = os.path.join(project_root, "results", "DRBN", "preview")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRBN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    dataset = DRBNDataset(
        os.path.join(project_root, "data", "Raw", "low"),
        os.path.join(project_root, "data", "Raw", "high")
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    start_epoch, num_epochs = 1, 500
    best_loss = float("inf")

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss, l1_accum, ssim_accum, color_accum = 0, 0, 0, 0

        for low, high in tqdm(loader, desc=f"DRBN Epoch {epoch}/{num_epochs}", ncols=100):
            low, high = low.to(device), high.to(device)
            out = model(low)

            l1 = F.l1_loss(out, high)
            ssim_l = ssim_loss(out, high)
            color_l = color_constancy_loss(out)
            loss = l1 + 0.3 * ssim_l + 0.05 * color_l

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            l1_accum += l1.item()
            ssim_accum += ssim_l.item()
            color_accum += color_l.item()

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        print(f"[Epoch {epoch:03d}/{num_epochs}] Total Loss: {epoch_loss:.4f}")
        print(f"  ▶ L1: {l1_accum/len(loader):.4f} | SSIM: {ssim_accum/len(loader):.4f} | Color: {color_accum/len(loader):.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"最佳模型更新 at epoch {epoch}, 儲存在: {best_path}")

        if epoch % 20 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_G_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint 儲存於: {ckpt_path}")

            model.eval()
            sample_low, sample_gt = next(iter(loader))
            with torch.no_grad():
                sample_out = model(sample_low.to(device))
            preview_path = os.path.join(preview_dir, f"epoch_{epoch}.png")
            save_image(torch.cat([sample_low, sample_out.cpu(), sample_gt], dim=0), preview_path, nrow=sample_low.size(0))
            print(f"預覽圖儲存於：{preview_path}")

    print("DRBN 訓練完成")


if __name__ == "__main__":
    train()
