import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from networks.retinexnet import UNetDecomNet, EnhanceNetDeep
from losses.retinexnet_loss import illumination_smoothness_loss, structure_loss, brightness_reg_loss, color_loss, laplacian_loss

def train_decomnet(decom_net, train_loader, save_root):
    optimizer = torch.optim.Adam(decom_net.parameters(), lr=1e-4)
    scaler = GradScaler()
    decom_ckpt_dir = os.path.join(save_root, "checkpoints")
    os.makedirs(decom_ckpt_dir, exist_ok=True)

    for epoch in range(1, 101):
        decom_net.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"[DecomNet Epoch {epoch}]")

        for low, high in loop:
            low, high = low.to(device), high.to(device)

            optimizer.zero_grad()
            with autocast():
                R_pred, I_pred = decom_net(low)
                loss_recon = F.l1_loss(R_pred * I_pred.expand_as(R_pred), low)
                loss_struct = structure_loss(R_pred, low)
                loss_smooth = illumination_smoothness_loss(I_pred, R_pred)  # ✅ 這裡修正成新版

                loss = loss_recon + 0.1 * loss_struct + 0.1 * loss_smooth

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(decom_net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[DecomNet Epoch {epoch}] avg_loss = {avg_loss:.6f}")

        if epoch % 5 == 0:
            torch.save(decom_net.state_dict(), f"{decom_ckpt_dir}/decom_epoch{epoch}.pth")
            print(f"Saved DecomNet checkpoint at Epoch {epoch}")

    torch.save(decom_net.state_dict(), f"{decom_ckpt_dir}/decom_final.pth")
    print("Saved final DecomNet model.")

def train_enhancenet(enhance_net, decom_net, train_loader, save_root):
    optimizer = torch.optim.Adam(enhance_net.parameters(), lr=1e-4)
    scaler = GradScaler()
    decom_ckpt_dir = os.path.join(save_root, "checkpoints")

    for epoch in range(1, 401):
        enhance_net.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"[EnhanceNet Epoch {epoch}]")

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
        print(f"[EnhanceNet Epoch {epoch}] avg_loss = {avg_loss:.6f}")

        if epoch % 5 == 0:
            torch.save(enhance_net.state_dict(), f"{decom_ckpt_dir}/enhance_epoch{epoch}.pth")
            print(f"Saved EnhanceNet checkpoint at Epoch {epoch}")

    torch.save(enhance_net.state_dict(), f"{decom_ckpt_dir}/enhance_final.pth")
    print("Saved final EnhanceNet model.")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_root = "C:/Users/hurry/OneDrive/桌面/LowLightEnhancement_PyTorch"
    low_dir = os.path.join(save_root, "data/Raw/low")
    high_dir = os.path.join(save_root, "data/Raw/high")

    from utils.dataset_retinexnet import LOLPatchDataset
    train_dataset = LOLPatchDataset(low_dir, high_dir, patch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    decom_net = UNetDecomNet().to(device)
    train_decomnet(decom_net, train_loader, save_root)

    decom_net.load_state_dict(torch.load(os.path.join(save_root, "checkpoints/decom_final.pth")))
    decom_net.eval()

    enhance_net = EnhanceNetDeep().to(device)
    train_enhancenet(enhance_net, decom_net, train_loader, save_root)
