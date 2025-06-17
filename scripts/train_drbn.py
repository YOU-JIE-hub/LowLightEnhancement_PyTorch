import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from networks.drbn import DRBN
from utils.dataset_drbn import DRBNDataset
from losses.drbn_loss import ssim_loss, color_constancy_loss  # 專用的 SSIM 與色彩一致性損失

from lpips import LPIPS   # 用於評估結構相似性

root_path = "C:/Users/hurry/OneDrive/桌面/LowLightEnhancement_PyTorch"
os.makedirs(f"{root_path}/checkpoints/DRBN", exist_ok=True)
os.makedirs(f"{root_path}/results/DRBN/preview", exist_ok=True)

model = DRBN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

lpips_loss = LPIPS(net='alex').cuda()

dataset = DRBNDataset(
    os.path.join(root_path, "data/Raw/low"),
    os.path.join(root_path, "data/Raw/high")
)
loader = DataLoader(dataset, batch_size=4, shuffle=True)  # 批次大小為 4

start_epoch, num_epochs = 1, 500
for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    epoch_loss, l1_accum, ssim_accum, color_accum = 0, 0, 0, 0

    for low, high in tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=100):
        low, high = low.cuda(), high.cuda()
        out = model(low)

        l1 = F.l1_loss(out, high)                  # 保圖像重建逼近 GT
        ssim_l = ssim_loss(out, high)              # 保留細節與結構一致性
        color_l = color_constancy_loss(out)        # 防止偏色
        loss = l1 + 0.3 * ssim_l + 0.05 * color_l

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        l1_accum += l1.item()
        ssim_accum += ssim_l.item()
        color_accum += color_l.item()

    scheduler.step()

    print(f"[Epoch {epoch:03d}/{num_epochs}] Total Loss: {epoch_loss:.4f}")
    print(f"  ▶ L1: {l1_accum/len(loader):.4f} | SSIM: {ssim_accum/len(loader):.4f} | Color: {color_accum/len(loader):.4f}")

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"{root_path}/checkpoints/DRBN/model_G_{epoch}.pth")

    if epoch % 20 == 0:
        model.eval()
        sample_low, sample_gt = next(iter(loader))
        with torch.no_grad():
            sample_out = model(sample_low.cuda())
        preview_path = f"{root_path}/results/DRBN/preview/epoch_{epoch}.png"
        save_image(torch.cat([sample_low, sample_out.cpu(), sample_gt], dim=0), preview_path, nrow=sample_low.size(0))
        print(f"預覽圖儲存於：{preview_path}")

print("Step1 訓練完成")
