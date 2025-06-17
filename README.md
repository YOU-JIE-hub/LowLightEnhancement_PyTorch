《低光影像增強之多模型整合實作與比較分析》（2025/04 – 2025/06）

本專案為元智大學電機工程學系乙組畢業專題，主題聚焦於「低光環境下的影像增強技術」，整合並實作多種低光影像增強模型，包含傳統演算法與深度學習方法，並進行訓練、測試與效果比較分析。

---

##  專案目標

- 重現並實作七種低光影像增強模型
- 修正與優化開源訓練不佳的模型架構
- 統一訓練與評估流程，進行主觀與客觀效果評估
- 實現完整從論文還原 → 模型訓練 → 結果比較的全流程

---

## 已實作模型

# 傳統方法
- LIME
- Retinex（MSRCR）
- Freq Filter（同態頻率濾波）

# 深度學習方法
- RetinexNet
- DRBN
- EnlightenGAN
- Zero-DCE

---

## 成果展示

- 客觀評估指標：PSNR、SSIM、NIQE

Model	            PSNR	      SSIM	         BRISQUE	    PI
FreqFilter	        8.418641058	  0.259216603	 16.92715251	13.46357625
LIME	            9.44970279	  0.291692113	 55.45676473	32.72838236
RetinexNet	        17.59915801	  0.717233259	 36.31658529	23.15829264
RetinexTraditional	13.82073052	  0.583829124	 17.26178385	13.63089193
ZeroDCE	            12.08034649	  0.457940512	 4.035050456	7.017525228
EnlightenGAN	    19.44944589	  0.714923392	 41.04030762	25.52015381
DRBN	            16.80202909	  0.747324044	 6.448982747	8.224491374

- 主觀視覺比較：多模型同圖對照強化結果

![比對圖1](./results/comparison1.png)
![比對圖2](./results/comparison2.png)
![比對圖3](./results/comparison3.png)
![比對圖4](./results/comparison4.png)

---

## 模型實作與重構概述

1.	RetinexNet

擴充六種自定義損失（illumination_smoothness、structure、color、brightness_reg、illumination_mean、laplacian 等）

導入 AMP 混合精度訓練與梯度裁剪以強化訓練穩定性與效率

模組化損失結構與訓練腳本，提升實驗靈活性與效能

2.	DRBN

拆解 RecursiveBlock 與 BandBlock 架構，重構模組化模型

損失函數組合：L1 + 0.3 × SSIM + 0.05 × ColorConstancy，並引入 LPIPS 與 VGG 評估指標

加入 CosineAnnealingLR 與結果預覽輸出機制，利於訓練穩定與結果監控

3.	EnlightenGAN

將 unpaired 訓練改為 paired（使用 LOL dataset），提升監督效果

改用 ImprovedUNet（Encoder-Decoder + skip connection），搭配 SpectralNorm 的 PatchGAN 判別器

整合 8 項損失（g/d_hinge、L1、LPIPS、VGG、SSIM、Color、Laplacian、TV）達到語意、色彩與結構同步優化

4.	Zero-DCE

還原完整 7 層結構輸出 24 通道 A 曲線參數，支援 apply_curve 應用邏輯

實作四種損失（ColorConstancy、SpatialConsistency、TotalVariation、Exposure）導向無監督亮度優化

封裝推論後處理與批次測試流程，支援 real-world 圖像增亮任務

5.	LIME

使用 NumPy 與 Scipy 高斯濾波模擬照明圖平滑過渡

全流程函式化為 enhance_lime()，支援任意 RGB 輸入與 uint8 輸出

改善照明圖估計與補光效果，提升細節自然性，適合即時應用部署

6.	RetinexTraditional（基於 MSR）

使用 PyTorch conv2d 完整實作 Gaussian 模糊流程，支援三組 sigma（15、80、250）

模組化 single_scale_retinex() 與 multi_scale_retinex()，支援 GPU batch 處理與實時應用

實作 log 安全處理與 min-max normalization，輸出穩定適用於部署與視覺化展示

7.	FreqFilter（頻域同態濾波）

還原 Homomorphic Filtering 全流程：log → fft → high-pass Gaussian filter → ifft → exp

對彩色圖像處理時進行 YCbCr 分離，僅處理亮度通道 Y，後合併回 RGB

加入預亮度補償（×1.5）與後處理 gamma(g=0.5, gain=1.4)，解決偏暗與 halo 問題

---

## 專案資料夾說明

LowLightEnhancement_PyTorch/
├──Final_Report.pdf # 畢業專題論文
├── losses/ # 所有訓練中使用的損失函數定義
├── networks/ # 各模型架構（RetinexNet、DRBN、Zero-DCE 等）
├── results/ # 各模型執行後之增強結果比較圖
├── scripts/ # 執行用的主要訓練與推論腳本（train.py, infer.py）
├── traditional/ # 傳統增強方法實作（LIME、MSR、同態濾波等）
├── utils/ # 輔助模組（資料分割、預處理、後處理）
└──  requirements.txt # 執行所需 Python 套件列表

---

## 補充資源（Google Drive）

因 GitHub 空間限制，完整資料集、模型權重與全部結果圖另提供於雲端：

👉 [點我前往下載區（Google Drive）](https://drive.google.com/file/d/1U7CDi63s4Z7tY5yxRXzrLGXeh6TnGDiz/view?usp=sharing)

├── checkpoints/ # 各模型訓練完成之權重檔 (.pt)
├── data/ # 原始與合成的低光影像資料集
├── debug/ # Colab 測試與除錯版本
├── debug_checkpoint # Colab 測試與除錯版本之權重檔 (.pt)
├── [model]/  [model_val] # 各模型訓練圖及結果測試圖，其中 post 資料夾表經過額外後處理的結果圖
└── Comparison # 結果比對圖及客觀指標統計圖

---

## 環境

- Python 3.10, PyTorch 2.x, OpenCV, NumPy
- Google Colab / VSCode / Linux CLI
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

> 本專案基於上述開源模型與論文，進行重構、修正與整合，並已於技術報告中完整說明引用來源。

---

## 🙋‍♂️ 作者資訊

- Email：h125872359@gmail.com
- 學歷：元智大學電機工程學系學士
- 技術專長：PyTorch、影像處理、模型重構、低光增強

> 本專案為本人畢業專題實作成果，歡迎交流！