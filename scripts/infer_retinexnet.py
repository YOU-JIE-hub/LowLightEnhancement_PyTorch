import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from networks.retinexnet import UNetDecomNet, EnhanceNetDeep
from losses.retinexnet_loss import (
    illumination_smoothness_loss, structure_loss,
    brightness_reg_loss, color_loss, laplacian_loss
)
from utils.dataset_retinexnet import LOLPatchDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_decomnet(decom_net, train_loader, save_root):
    optimizer = torch.optim.Adam(decom_net.parameters(), lr=1e-4)
    scaler = GradScaler()
    decom_ckpt_dir = os.path.join(save_root, "checkpoints", "RetinexNet")
    os.makedirs(decom_ckpt_dir, exist_ok=True)
    num_epochs = 100
    best_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        decom_net.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"DecomNet Epoch {epoch}/{num_epochs}", ncols=100)

        for low, high in loop:
            low, high = low.to(device), high.to(device)

            optimizer.zero_grad()
            with autocast():
                R_pred, I_pred = decom_net(low)
                loss_recon = F.l1_loss(R_pred * I_pred.expand_as(R_pred), low)
                loss_struct = structure_loss(R_pred, low)
                loss_smooth = illumination_smoothness_loss(I_pred, R_pred)
                loss = loss_recon + 0.1 * loss_struct + 0.1 * loss_smooth

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(decom_net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[DecomNet Epoch {epoch:03d}/1] Total Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(decom_ckpt_dir, "decom_best.pth")
            torch.save(decom_net.state_dict(), best_path)
            print(f"最佳 DecomNet 更新於 epoch {epoch}，儲存於: {best_path}")

    final_path = os.path.join(decom_ckpt_dir, "decom_final.pth")
    torch.save(decom_net.state_dict(), final_path)
    print(f"儲存最終 DecomNet model at: {final_path}")

def train_enhancenet(enhance_net, decom_net, train_loader, save_root):
    optimizer = torch.optim.Adam(enhance_net.parameters(), lr=1e-4)
    scaler = GradScaler()
    enhance_ckpt_dir = os.path.join(save_root, "checkpoints", "RetinexNet")
    preview_dir = os.path.join(save_root, "results", "RetinexNet", "preview")
    os.makedirs(enhance_ckpt_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    num_epochs = 400
    for epoch in range(1, num_epochs + 1):
        enhance_net.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"EnhanceNet Epoch {epoch}/{num_epochs}", ncols=100)

        for low, high in loop:
            low, high = low.to(device), high.to(device)
            with torch.no_grad():
                R_low, I_low = decom_net(low)

            optimizer.zero_grad()
            with autocast():
                I_new, color_map = enhance_net(R_low, I_low)
                enhanced = (R_low + color_map) * I_new.expand_as(R_low)
                enhanced = torch.clamp(enhanced, 0, 1)

                relight = F.l1_loss(enhanced, high)
                bright = brightness_reg_loss(enhanced, lower=0.4, upper=0.7)
                colorv = color_loss(enhanced, high)
                percep = structure_loss(enhanced, high)
                lap = laplacian_loss(enhanced)
                loss = 1.0 * relight + 10.0 * bright + 1.0 * colorv + 0.2 * percep + 0.1 * lap

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(enhance_net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[EnhanceNet Epoch {epoch:03d}/400] Total Loss: {avg_loss:.6f}")

        if epoch % 20 == 0:
            ckpt_path = os.path.join(enhance_ckpt_dir, f"enhance_epoch{epoch}.pth")
            torch.save(enhance_net.state_dict(), ckpt_path)
            print(f"Checkpoint 儲存於: {ckpt_path}")

            enhance_net.eval()
            sample_low, sample_gt = next(iter(train_loader))
            with torch.no_grad():
                R_low, I_low = decom_net(sample_low.to(device))
                I_new, color_map = enhance_net(R_low, I_low)
                enhanced = (R_low + color_map) * I_new.expand_as(R_low)
                enhanced = torch.clamp(enhanced, 0, 1).cpu()
            os.makedirs(preview_dir, exist_ok=True)
            preview_path = os.path.join(preview_dir, f"epoch_{epoch}.png")
            save_image(torch.cat([sample_low, enhanced, sample_gt], dim=0), preview_path, nrow=sample_low.size(0))
            print(f"預覽圖儲存於：{preview_path}")#　第 1 列：低光原圖，第 2 列：RetinexNet 增強結果，第 3 列：原高光圖
            enhance_net.train()

    final_path = os.path.join(enhance_ckpt_dir, "enhance_final.pth")
    torch.save(enhance_net.state_dict(), final_path)
    print(f"儲存最終 EnhanceNet model at: {final_path}")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_root = project_root

    low_dir = os.path.join(save_root, "data", "Raw", "low")
    high_dir = os.path.join(save_root, "data", "Raw", "high")
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)

    train_dataset = LOLPatchDataset(low_dir, high_dir, patch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    decom_net = UNetDecomNet().to(device)
    train_decomnet(decom_net, train_loader, save_root)

    decom_net.load_state_dict(torch.load(os.path.join(save_root, "checkpoints", "RetinexNet", "decom_final.pth")))
    decom_net.eval()

    enhance_net = EnhanceNetDeep().to(device)
    train_enhancenet(enhance_net, decom_net, train_loader, save_root)

    print("RetinexNet 訓練完成")

if __name__ == "__main__":
    main()
