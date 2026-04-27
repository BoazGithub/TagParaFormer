# TagParaFormer 🛣️

**Cross-Hybrid Attention Learning Framework for Topology-Aware Road Network Extraction**

> Boaz Mwubahimana · Dingruibo Miao · Yan Jianguo · Swalpa Kumar Roy · Le Ma · Remy Dukundane · Xiao Huang · Ruisheng Wang
>
> *State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing (LIESMARS), Wuhan University*

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-orange)](https://github.com/BoazGithub/TagParaFormer)
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/BoazGithub/TagParaFormer)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

Road network extraction from very high-resolution (VHR) remote sensing imagery requires both geometric precision and topological connectivity — a challenge conventional CNNs and transformers struggle to address because they treat road extraction as per-pixel classification rather than graph reconstruction.

**TagParaFormer** reformulates road extraction as a **topology-preserving representation learning** problem. It integrates:

- A **dual-branch encoder** (CNN + 3-layer ViT) for joint local–global feature extraction
- An **Attention Node Graphical (ANG)** module for explicit node-level junction modeling
- A **Road Refinement Backbone (RRB)** for edge-level structural correction
- A **Bayesian Parzen-Tree Estimator (PTE)** to auto-tune topology-sensitive hyperparameters (α, β, λ)
- **TagLoss**: BCE + Distributional Variance Adjustment (DVA) + skeleton topology loss

---

## 🏆 Key Results

| Dataset | mIoU | OA | F1 | BF1 | CR | GFR | Params |
|---|---|---|---|---|---|---|---|
| **DeepGlobe** | **91.96%** | **96.83%** | **93.42%** | **93.6%** | **94.5%** | **92.8%** | **11.32M** |
| Massachusetts | 88.02% | 96.70% | 81.0% | 79.0% | — | — | 11.32M |
| SpaceNet | 89.15% | 96.60% | 84.33% | 82.0% | — | — | 11.32M |

**vs. Competing Methods (DeepGlobe)**:
- ↑ **4.53% mIoU** over SAM-Road (67M params)
- ↓ **47% fragmentation** vs. RNGDet++
- ↓ **32% error reduction** vs. AttUNet
- **5–6× fewer parameters** than transformer-heavy baselines

---

## 🏗️ Architecture

```
Input (B, 3, H, W)
       │
       ├──────────────────────────────────┐
       ▼                                  ▼
  CNN Branch                         ViT Branch
  (ResNet-34 style,                  (3-layer MHSA,
   8 residual blocks,                 patch size 32,
   coordinate attention GARL)         embed_dim=256)
       │                                  │
       └──────────┬───────────────────────┘
                  ▼
          ANG Module
          (Coordinate attention → node identification)
          (K=64 learnable node queries → graph adjacency)
                  │
                  ▼
          RRB Module
          (Gated conv + cross-branch attention + dilated residuals)
                  │
                  ▼
          Adaptive Fusion  α·F_cnn + β·F_vit
          (PTE-optimized: α*=0.6, β*=0.4)
                  │
                  ▼
          Segmentation Head → (B, 1, H, W)
          ↓ upsample to (B, 1, H, W)
          Road Prediction Map
```

### Module Descriptions

| Module | Function | Key Equations |
|---|---|---|
| **CNN Branch** | Local geometric features via 8 residual blocks + UNet skip connections | H_{i,c}(X) = F_{i,c}(X) ⊙ M_{i,c}(X) + X |
| **ViT Branch** | Long-range connectivity via 3-layer multi-head self-attention | A_ij = softmax(Q_i K_j^T / √d_k) |
| **ANG** | Node-level junction modeling via coordinate attention + K=64 graph node queries | Z_c^h(h) = 1/W · Σ_i X_c(h,i) |
| **RRB** | Edge-level structural correction via gated conv + dilated residuals (1,2,4) | F_refined = Gate(F_cnn) ⊙ Attention(F_vit, F_cnn) |
| **PTE** | Bayesian TPE optimization of (α, β, λ) | x* = argmax_x α(x) = E[max(0, f(x*) − f(x))] |
| **TagLoss** | BCE + Poisson DVA + skeleton topology | L_total = L_BCE + L_DVA + L_DVA' |

---

## 📂 Repository Structure

```
TagParaFormer/
├── models/
│   ├── tagparaformer.py     ← Main model class
│   ├── cnn_branch.py        ← ResNet-34 encoder-decoder with GARL
│   └── vit_branch.py        ← 3-layer Vision Transformer
│
├── modules/
│   ├── ang.py               ← Attention Node Graphical module
│   ├── rrb.py               ← Road Refinement Backbone
│   ├── fusion.py            ← Adaptive α/β weighted fusion
│   ├── losses.py            ← TagLoss (BCE + DVA + topology)
│   └── pte.py               ← Bayesian Parzen-Tree Estimator
│
├── utils/
│   └── metrics.py           ← OA, F1, mIoU, BF1, CP, CR, GFR, JA
│
├── datasets/
│   └── road_datasets.py     ← DeepGlobe / Massachusetts / SpaceNet / WHU
│
├── train/
│   └── train.py             ← Training loop (AMP, cosine LR, DDP-ready)
│
├── inference/
│   └── predict.py           ← Single-image & tiled sliding-window inference
│
├── scripts/
│   ├── evaluate.py          ← Full test-set evaluation
│   ├── run_pte.py           ← Standalone Bayesian PTE search
│   ├── visualize.py         ← Feature maps, radar, PR/ROC curves
│   └── spacenet_prepare.py  ← SpaceNet GeoJSON → binary mask conversion
│
├── configs/
│   ├── deepglobe.yaml       ← PTE-optimized config (α*=0.6, β*=0.4, λ*=0.3)
│   └── massachusetts.yaml
│
├── requirements.txt
└── setup.py
```

---

## ⚙️ Installation

```bash
# 1. Clone
git clone https://github.com/BoazGithub/TagParaFormer
cd TagParaFormer

# 2. Create environment (Python 3.8+)
conda create -n tagparaformer python=3.10
conda activate tagparaformer

# 3. Install PyTorch (adjust CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install as package (for imports)
pip install -e .
```

---

## 🗂️ Dataset Preparation

### Expected directory structure

```
data/
├── deepglobe/
│   ├── train/
│   │   ├── images/   # RGB .jpg  (0.5m)
│   │   └── masks/    # Binary .png (0=bg, 255=road)
│   ├── val/
│   └── test/
├── massachusetts/
│   ├── train/ val/ test/  (same structure)
├── spacenet/
│   └── train/ val/ test/
└── whu/
    └── train/ val/ test/
```

### SpaceNet GeoJSON → PNG conversion

```bash
python scripts/spacenet_prepare.py \
    --geojson_dir  data/spacenet/raw/geojson \
    --image_dir    data/spacenet/raw/images \
    --out_mask_dir data/spacenet/train/masks \
    --buffer 2
```

---

## 🚀 Training

### Step 1 — Initial training

```bash
python train/train.py --config configs/deepglobe.yaml
```

Checkpoint saved to `checkpoints/deepglobe/best.pth`.

### Step 2 — Bayesian PTE hyperparameter search

Run after initial training to find topology-optimal (α*, β*, λ*):

```bash
python scripts/run_pte.py \
    --config     configs/deepglobe.yaml \
    --checkpoint checkpoints/deepglobe/best.pth \
    --trials     100
```

This updates `configs/deepglobe.yaml` with optimized weights.

### Step 3 — Retrain with PTE-optimized hyperparameters

```bash
python train/train.py --config configs/deepglobe.yaml
```

### Training configuration

| Parameter | DeepGlobe | Notes |
|---|---|---|
| Batch size | 16 | |
| Epochs | 200 | |
| LR | 1e-3 | AdamW, backbone LR × 0.1 |
| LR schedule | Cosine + 10-ep warmup | |
| AMP | FP16 | GradScaler |
| α* (CNN) | 0.6 | PTE-optimized |
| β* (ViT) | 0.4 | PTE-optimized |
| λ* (DVA) | 0.3 | PTE-optimized |

---

## 📊 Evaluation

```bash
python scripts/evaluate.py \
    --config     configs/deepglobe.yaml \
    --checkpoint checkpoints/deepglobe/best.pth \
    --split      test \
    --threshold  0.3 \
    --save_preds
```

Outputs all 11 metrics: OA, P, R, F1, IoU, mIoU, BF1, CP, CR, GFR, JA.

---

## 🔍 Inference

```bash
# Single image
python inference/predict.py \
    --checkpoint checkpoints/deepglobe/best.pth \
    --input      my_image.png \
    --output     road_pred.png \
    --threshold  0.3 \
    --visualize

# Large aerial scene (tiled sliding window, 32px overlap)
python inference/predict.py \
    --checkpoint checkpoints/deepglobe/best.pth \
    --input      large_scene.tif \
    --output     scene_pred.png \
    --tiled
```

---

## 📐 Ablation Study (Table IV)

| Configuration | mIoU | OA | F1 |
|---|---|---|---|
| CNN only | 82.34 | 94.56 | 88.76 |
| + ViT branch | 85.67 | 95.34 | 90.23 |
| + ANG module | 88.34 | 96.12 | 91.87 |
| + RRB module | 89.23 | 96.34 | 92.34 |
| + TagLoss | 90.87 | 96.51 | 92.98 |
| + PTE optimization | 91.56 | 96.71 | 93.24 |
| **TagParaFormer (full)** | **91.96** | **96.83** | **93.42** |

---

## 📜 Citation

```bibtex
@article{mwubahimana2026tagparaformer,
  title   = {TagParaFormer: Cross-Hybrid Attention Learning Framework for
             Topology-Aware Road Network Extraction},
  author  = {Mwubahimana, Boaz and Miao, Dingruibo and Jianguo, Yan and
             Roy, Swalpa Kumar and Ma, Le and Dukundane, Remy and
             Huang, Xiao and Wang, Ruisheng},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {Under Review}
}
```


---

## 📧 Contact

- **Boaz Mwubahimana** — m.boaz@whu.edu.cn (LIESMARS, Wuhan University)
- **Yan Jianguo** — jgyan@whu.edu.cn *(corresponding)*
- **Dingruibo Miao** — miaodrb@whu.edu.cn *(co-corresponding)*

---

## 📄 License

MIT License. Dataset usage is subject to the respective challenge/provider licenses:
- DeepGlobe: [DeepGlobe Challenge Terms](http://deepglobe.org/)
- SpaceNet: [SpaceNet License](https://spacenet.ai/)
- Massachusetts: University of Toronto
- WHU: Wuhan University
