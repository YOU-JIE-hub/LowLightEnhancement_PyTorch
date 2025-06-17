import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 導入模型與資料集、曲線應用函式
from networks.zero_dce import ZeroDCE
from utils.dataset_lol_dce import LOLPairedDataset
from utils.dce_utils import apply_curve

# TV Loss：鼓勵生成的增強曲線在空間上平滑，防止不連續的強化效果
def loss_tv(A):
    h_x = A.size()[2]
    w_x = A.size()[3]
    count_h = (A[:, :, 1:, :] - A[:, :, :h_x-1, :]).abs().sum()
    count_w = (A[:, :, :, 1:] - A[:, :, :, :w_x-1]).abs().sum()
    return (count_h + count_w) / (A.size()[0] * A.size()[1] * h_x * w_x)

# 保持影像三個通道的顏色平衡，減少色偏
def loss_color(image):
    mean_rgb = torch.mean(image, dim=[2,3], keepdim=True)  # 每張圖平均顏色值
    mr, mg, mb = mean_rgb[:,0,:,:], mean_rgb[:,1,:,:], mean_rgb[:,2,:,:]
    drg = torch.pow(mr - mg, 2)
    drb = torch.pow(mr - mb, 2)
    dgb = torch.pow(mb - mg, 2)
    return torch.mean(torch.sqrt(drg + drb + dgb + 1e-6))

# 讓增強後的影像曝光平均接近 mean_val

def loss_exposure(image, mean_val=0.6):
    mean = torch.mean(image, dim=[2,3])
    return torch.mean(torch.abs(mean - mean_val))

# 原圖與增強圖保持整體結構一致

def loss_spatial_consistency(org, enhanced):
    return F.l1_loss(org, enhanced)

batch_size = 8
num_epochs = 100
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lowlight_dir = r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\data\Raw\low"
highlight_dir = r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\data\Raw\high"

train_dataset = LOLPairedDataset(lowlight_dir, highlight_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = ZeroDCE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    loop = tqdm(train_loader, leave=True)

    for i, (low_img, _) in enumerate(loop):
        low_img = low_img.to(device)
        optimizer.zero_grad()

        A = model(low_img)  # 預測曲線參數 A
        enhanced = apply_curve(low_img, A)  # 應用曲線增強影像

        # 損失函數組合
        loss = loss_color(enhanced) + \
               loss_exposure(enhanced) + \
               loss_tv(A)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss/len(train_loader):.4f}")

    save_dir = r"C:\Users\hurry\OneDrive\桌面\LowLightEnhancement_PyTorch\checkpoints\ZeroDCE"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"zero_dce_epoch{epoch+1}.pth"))
