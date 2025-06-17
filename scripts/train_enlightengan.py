import os
import random
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from networks.enlightengan_unet import ImprovedUNet, Discriminator, ReplayBuffer
from losses.enlightengan_loss import *
from utils.val_dataset import UnpairedEnlightenDataset, ValDataset
import lpips
from torchvision.models import vgg16, VGG16_Weights

if __name__ == '__main__':
    CKPT_DIR = "C:/Users/hurry/OneDrive/桌面/LowLightEnhancement_PyTorch/checkpoints/EnlightenGAN"
    PREVIEW_DIR = "C:/Users/hurry/OneDrive/桌面/LowLightEnhancement_PyTorch/results/EnlightenGAN/preview"
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = ImprovedUNet().to(device)
    D = Discriminator().to(device)
    fake_pool = ReplayBuffer()

    # 使用最新的 VGG 權重
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    vgg_loss_fn = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval().to(device)
    for param in vgg_loss_fn.parameters():
        param.requires_grad = False  # 固定 VGG 權重

    optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    schedG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=100)
    schedD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, T_max=100)

    loss_weights = {
        'w_adv': 1.0,
        'w_pix': 3.0,
        'w_lpips': 1.0,
        'w_vgg': 0.5,
        'w_ssim': 1.0,
        'w_color': 1.0,
        'w_lap': 1.0,
        'w_tv': 0.5
    }

    low_dir = "C:/Users/hurry/OneDrive/桌面/LowLightEnhancement_PyTorch/data/Raw/low"
    high_dir = "C:/Users/hurry/OneDrive/桌面/LowLightEnhancement_PyTorch/data/Raw/high"
    train_ds = UnpairedEnlightenDataset(low_dir, high_dir)
    val_ds = ValDataset(low_dir, high_dir)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    for epoch in range(1, 101):
        stats = {k: 0.0 for k in ['D', 'adv', 'pix', 'lpips', 'vgg', 'ssim', 'color', 'lap', 'tv', 'G']}
        G.train()
        D.train()

        for low, high in train_loader:
            low, high = low.to(device), high.to(device)
            fake = G(low)
            # 訓練判別器 D
            optD.zero_grad()
            real_pred = D(high)
            fake_pred = D(fake_pool.push_and_pop(fake.detach()))
            loss_D = d_hinge_loss(real_pred, fake_pred)
            loss_D.backward()
            optD.step()

            # 訓練生成器 G
            optG.zero_grad()
            fp_g = D(fake)
            loss_adv = g_hinge_loss(fp_g) * loss_weights['w_adv']
            loss_pix = F.l1_loss(fake, high) * loss_weights['w_pix']
            loss_lp = lpips_fn(fake, high).mean() * loss_weights['w_lpips']
            loss_vgg = F.mse_loss(vgg_loss_fn(fake), vgg_loss_fn(high)) * loss_weights['w_vgg']
            loss_ssim = ssim_loss(fake, high) * loss_weights['w_ssim']
            loss_col = color_loss(fake) * loss_weights['w_color']
            loss_lap = lap_loss(fake, high) * loss_weights['w_lap']
            loss_tv_ = tv_loss(fake) * loss_weights['w_tv']
            loss_G = loss_adv + loss_pix + loss_lp + loss_vgg + loss_ssim + loss_col + loss_lap + loss_tv_
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(),  0.5)
            optG.step()

            stats['D'] += loss_D.item()
            stats['adv'] += loss_adv.item()
            stats['pix'] += loss_pix.item()
            stats['lpips'] += loss_lp.item()
            stats['vgg'] += loss_vgg.item()
            stats['ssim'] += loss_ssim.item()
            stats['color'] += loss_col.item()
            stats['lap'] += loss_lap.item()
            stats['tv'] += loss_tv_.item()
            stats['G'] += loss_G.item()

        schedG.step()
        schedD.step()
        avg_stats = {k: v / len(train_loader) for k, v in stats.items()}
        print(f"[Epoch {epoch}/100] " + " ".join(f"{k}:{v:.4f}" for k, v in avg_stats.items()))

        if epoch % 5 == 0:
            torch.save(G.state_dict(), os.path.join(CKPT_DIR, f"G_epoch_{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(CKPT_DIR, f"D_epoch_{epoch}.pth"))

            G.eval()
            samples = []
            with torch.no_grad():
                for _ in range(3):
                    idx = random.randint(0, len(val_ds) - 1)
                    low, high = val_ds[idx]
                    fake = G(low.unsqueeze(0).to(device)).squeeze().cpu()
                    samples.extend([high, low, fake])
                grid = make_grid(samples, nrow=3)
            G.train()
