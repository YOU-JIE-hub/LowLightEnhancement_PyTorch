## 《低光影像增強之多模型整合實作與比較分析》
###（2025/04 – 2025/06）

---

本專案為元智大學電機工程學系乙組畢業專題，主題聚焦於「低光環境下的影像增強技術」，整合並實作多種低光影像增強模型，包含傳統演算法與深度學習方法，並進行訓練、測試與效果比較分析。
```
git clone https://github.com/YOU-JIE-hub/LowLightEnhancement_PyTorch.git
cd LowLightEnhancement_PyTorch
```
安裝依賴
```
pip install -r requirements.txt
```
執行範例(換成對應執行檔案)
```
python -m scripts.train_drbn
```
執行注意事項

- 所有腳本請在根目錄, 以 python -m scripts.xxx 格式執行
- 若要執行推論腳本須確保 checkpoints/ 有對應權重檔, 可從下方雲端連結下載資料夾 checkpoints 然後直接放在根目錄下
- 若要執行訓練腳本須確保 data/Raw/low、data/Raw/high 皆有資料, 可從下方雲端連結下載資料夾 data 然後直接放在根目錄下
- 訓練腳本每 20 epoch 會產生對應預覽圖於 results/[ModelName]/preview/
- 模型訓練皆會自動儲存至 checkpoints/{ModelName}/ 

---

##  專案目標

- 重現並實作七種低光影像增強模型
- 修正與優化開源訓練不佳的模型架構
- 統一訓練與評估流程，進行主觀與客觀效果評估
- 實現完整從論文還原 → 模型訓練 → 結果比較的全流程

---

## 已實作模型
### 傳統方法
- LIME
- Retinex（MSRCR）
- Freq Filter（同態頻率濾波）

### 深度學習方法
- RetinexNet
- DRBN
- EnlightenGAN
- Zero-DCE

---

## 成果展示

- 客觀評估指標：PSNR、SSIM、NIQE

| Model              | PSNR    | SSIM    | BRISQUE | PI      |
|--------------------|---------|---------|---------|---------|
| FreqFilter         | 8.42    | 0.259   | 16.93   | 13.46   |
| LIME               | 9.45    | 0.292   | 55.46   | 32.73   |
| RetinexNet         | 17.60   | 0.717   | 36.32   | 23.16   |
| RetinexTraditional | 13.82   | 0.584   | 17.26   | 13.63   |
| ZeroDCE            | 12.08   | 0.458   | 4.04    | 7.02    |

- 主觀視覺比較：多模型同圖對照強化結果

![比對圖1](https://github.com/YOU-JIE-hub/LowLightEnhancement_PyTorch/blob/main/example_results/comparison1.jpg)

![比對圖2](https://github.com/YOU-JIE-hub/LowLightEnhancement_PyTorch/blob/main/example_results/comparison2.jpg)

![比對圖3](https://github.com/YOU-JIE-hub/LowLightEnhancement_PyTorch/blob/main/example_results/comparison3.jpg)

![比對圖4](https://github.com/YOU-JIE-hub/LowLightEnhancement_PyTorch/blob/main/example_results/comparison4.jpg)

---

## 模型實作與重構概述

### 1.	RetinexNet

擴充六種自定義損失（illumination_smoothness、structure、color、brightness_reg、illumination_mean、laplacian 等）

導入 AMP 混合精度訓練與梯度裁剪以強化訓練穩定性與效率

模組化損失結構與訓練腳本，提升實驗靈活性與效能

### 2.	DRBN

拆解 RecursiveBlock 與 BandBlock 架構，重構模組化模型

損失函數組合：L1 + 0.3 × SSIM + 0.05 × ColorConstancy，並引入 LPIPS 與 VGG 評估指標

加入 CosineAnnealingLR 與結果預覽輸出機制，利於訓練穩定與結果監控

本專題僅參考原模型STEP1部分

### 3.	EnlightenGAN

將 unpaired 訓練改為 paired（使用 LOL dataset），提升監督效果

改用 ImprovedUNet（Encoder-Decoder + skip connection），搭配 SpectralNorm 的 PatchGAN 判別器

整合 8 項損失（g/d_hinge、L1、LPIPS、VGG、SSIM、Color、Laplacian、TV）達到語意、色彩與結構同步優化

### 4.	Zero-DCE

還原完整 7 層結構輸出 24 通道 A 曲線參數，支援 apply_curve 應用邏輯

實作四種損失（ColorConstancy、SpatialConsistency、TotalVariation、Exposure）導向無監督亮度優化

封裝推論後處理與批次測試流程，支援 real-world 圖像增亮任務

### 5.	LIME

使用 NumPy 與 Scipy 高斯濾波模擬照明圖平滑過渡

全流程函式化為 enhance_lime()，支援任意 RGB 輸入與 uint8 輸出

改善照明圖估計與補光效果，提升細節自然性，適合即時應用部署

### 6.	RetinexTraditional（基於 MSR）

使用 PyTorch conv2d 完整實作 Gaussian 模糊流程，支援三組 sigma（15、80、250）

模組化 single_scale_retinex() 與 multi_scale_retinex()，支援 GPU batch 處理與實時應用

實作 log 安全處理與 min-max normalization，輸出穩定適用於部署與視覺化展示

### 7.	FreqFilter（頻域同態濾波）

還原 Homomorphic Filtering 全流程：log → fft → high-pass Gaussian filter → ifft → exp

對彩色圖像處理時進行 YCbCr 分離，僅處理亮度通道 Y，後合併回 RGB

加入預亮度補償（×1.5）與後處理 gamma(g=0.5, gain=1.4)，解決偏暗與 halo 問題

---

## 專案資料夾說明
```
LowLightEnhancement_PyTorch/
├── Final_Report.pdf              # 畢業專題論文
├── losses/                       # 所有訓練中使用的損失函數定義
├── networks/                     # 各模型架構（RetinexNet、DRBN、Zero-DCE 等）
├── results/                      # 各模型執行後之增強結果比較圖
├── scripts/                      # 訓練、推論、比對與指標評估腳本
│   ├── train_xxx.py              # 各模型訓練腳本（Ex: train_drbn.py）
│   ├── infer_xxx.py              # 各模型推論腳本（Ex: infer_retinexnet.py）
│   ├── visualize_comparison.py   # 輸出多模型對照圖
│   └── evaluate_all_models.py    # 批次計算所有模型的 PSNR / SSIM 等指標
├── traditional/                  # 傳統增強方法實作（LIME、MSR、同態濾波等）
├── utils/                        # 資料集切分、資料讀取、後處理模組
└── requirements.txt              # Python 套件需求清單
```
---

## 補充資源（Google Drive）

因 GitHub 空間限制，完整資料集、模型權重、測試版本、全部訓練結果圖及驗證結果圖另提供於雲端連結：

👉 [點我前往下載區（Google Drive）](https://drive.google.com/drive/folders/1ONZraTVOyk__ASMSUu8K3sL6q_jefm26?usp=sharing)

Google Drive 資料夾結構如下：
```
├── checkpoints/             # 各模型訓練完成之權重檔 (.pth)
├── data/                    # 原始與合成的低光影像資料集（LOL dataset）
├── debug/                   # Colab 測試與除錯版本
├── debug_checkpoint/        # 測試版本用的 checkpoint
├── [model]/                 # 各模型的訓練過程輸出圖
├── └── post/                # 經後處理（ex: gamma 校正）的最終輸出圖
├── [model_val]/             # 各模型的測試結果圖（val set）
│   └── post/                # 經後處理（ex: gamma 校正）的最終輸出圖
└── Comparison/              # 多模型對照圖與客觀指標統計（PSNR/SSIM/NIQE）
```
---

## 模型儲存和載入格式

### EnlightenGAN
- Train: saves complete checkpoint:
```
  checkpoint = {
    "epoch": epoch,
    "G_state": G.state_dict(),
    "D_state": D.state_dict(),
    "optG": optG.state_dict(),
    "optD": optD.state_dict(),
    "loss": avg_stats
  }
  torch.save(checkpoint, "checkpoint_epoch_{epoch}.pth")
```
- Infer: auto-fallback load:
```
ckpt = torch.load(ckpt_path, map_location=device)
if "G_state" in ckpt:
    model.load_state_dict(ckpt["G_state"])
else:
    model.load_state_dict(ckpt)
model.eval()
```
### RetinexNet / ZeroDCE / DRBN
- Train: saves only model weights (.pth)
- Infer: single-command load:
```
model.load_state_dict(torch.load("xxx.pth", map_location=device))
```
---

## 模型評估腳本使用說明
### 1. 執行evaluate_all_models時
#### 輸入要求
Ground Truth 路徑：
```
data/Raw/high_val/
```
各模型推論結果需放於：
```
results/{模型名_val}/{對應檔名}.png
```
#### 輸出內容
評估指標表格（CSV）：
```
results/Comparison/quality_val.csv
```
雷達圖可視化：
```
results/Comparison/quality_val_radar.png
results/Comparison/radar_val_group1_dl.png
results/Comparison/radar_val_group2_traditional.png
```
### 2. 若無法生成雷達圖：quality_val.csv 欠缺必要欄位或為空
請確認以下事項：
1. 已先完成所有模型的推論。
2. 所有預測圖檔皆已存於對應資料夾，檔名與 Ground Truth 一致。
3. Ground Truth 路徑 data/Raw/high_val/ 下有對應圖片。
   
### 3. 執行 visualize_comparison 前需確保已執行過7個模型的推論產出結果圖
---

## 環境

- Python 3.10, PyTorch 2.x
- OpenCV, PIL, numpy, tqdm
- Google Colab / VSCode + CUDA
- 資料集：LOL Dataset

---

## 參考資料

1.	Wei, C., Wang, W., Yang, W., & Liu, J. (2018). Deep Retinex Decomposition for Low-Light Enhancement. In Proceedings of the British Machine Vision Conference (BMVC).
Yonghui Wang.2022.Deep Retinex Decomposition for Low-Light Enhancement, BMVC'18.
https://github.com/harrytea/RetinexNet.
(2025).
2.	Li, C., Gu, S., Liu, J., & Loy, C. C. (2020). Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
Li-Chongyi.2024.Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement.
https://github.com/Li-Chongyi/Zero-DCE.
(2025).
3.	Jiang, Y., Gong, X., Liu, D., Cheng, Y., Fang, C., Shen, X., & Yang, J. (2021). EnlightenGAN: Deep Light Enhancement Without Paired Supervision. IEEE Transactions on Image Processing, 30, 2340–2349.
VITA.2019.EnlightenGAN: Deep Light Enhancement without Paired Supervision.
https://github.com/VITA-Group/EnlightenGAN.
(2025).
4.	Yang, W., Wang, S., Fang, Y., Wang, Y., & Liu, J. (2020). From fidelity to perceptual quality: A semi-supervised approach for low-light image enhancement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 3063–3072).
flyywh.2022.From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement (CVPR'2020).
https://github.com/flyywh/CVPR-2020-Semi-Low-Light
(2025).
5.	Guo, X., Li, Y., & Ling, H. (2017). LIME: Low-Light Image Enhancement via Illumination Map Estimation. IEEE Transactions on Image Processing, 26(2), 982–993.
aeinrw.2021.Low-light Image Enhancement.
https://github.com/aeinrw/LIME.
(2025).
6.	Land, E. H., & McCann, J. J. (1971). Lightness and Retinex theory. Journal of the Optical Society of America, 61(1), 1–11.
dongb5.2023.Multiscale retinex with color restoration.
https://github.com/dongb5/Retinex.
(2025).
7.	Oppenheim, A. V., Schafer, R. W., & Stockham, T. G. (1968). Nonlinear Filtering of Multiplied and Convolved Signals. Proceedings of the IEEE, 56(8), 1264–1291.

本專案基於上述開源模型與論文，進行重構、修正與整合。

---

## 🙋‍♂️ 作者資訊

- Email：h125872359@gmail.com
- 學歷：元智大學電機工程學系學士
- 技術專長：PyTorch、影像處理、模型重構、低光增強

本專案為本人畢業專題實作成果，歡迎交流！
