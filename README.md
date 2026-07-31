# GAP-Track: Bridging the Resolution Gap for Cross-Resolution RGBT Tracking

This repository contains the official implementation of **GAP-Track**, an efficient RGBT tracking framework that bridges the resolution gap by enabling high-precision tracking of **low-resolution inputs**. It is built on top of the **CKD** baseline (Coupled Knowledge Distillation, ACM MM 2024).

## 🎯 Motivation

RGBT tracking on edge devices is often constrained by limited computational power and data transmission bandwidth, necessitating the use of **low-resolution inputs** in real-world deployments. However, such resolution reduction typically leads to severe semantic loss and performance degradation.

**GAP-Track** tackles this challenge by keeping the precision of high-resolution tracking while operating at much lower input resolutions (1/2× and 1/4×).

## 🔥 Key Contributions

1. **Hierarchical Knowledge Distillation (HKD)**  
   We guide the low-resolution student with **multi-level teacher supervision**, distilling knowledge from a high-resolution teacher across multiple feature levels to preserve semantic richness in the student.

2. **Auxiliary Generative Reconstruction Module with Random Masking**  
   To recover the missing semantic details, we introduce a generative reconstruction module with a **random masking strategy**. It strengthens the student's feature representation by forcing the backbone to recover fine-grained structural information from sparse pixels **during training only**.

3. **Polar-Geometric Sensitivity Loss (PGS-Loss)**  
   We decouple bounding box regression into a **polar coordinate system**, providing higher sensitivity to **center-point drift and shape deformations** — effectively mitigating localization ambiguities in extremely low-resolution scenarios.

> 💡 **Efficiency**: The teacher and the auxiliary generative module are used **only in the training stage**. At inference, only the low-resolution student branch and its tracking head are required, so the computational cost stays as low as the baseline.

## 📊 Results

### RGBT234
Bold numbers indicate the best performance.

| Algorithm | Source | Full Resolution (PR/SR) | Half Res 1/2 (PR/SR) | Quart. Res 1/4 (PR/SR) |
|-----------|--------|------------------------|----------------------|------------------------|
| MPLT  | arXiv'23 | 88.4 / 65.7 | 81.7 / 59.0 | 75.1 / 53.1 |
| CKD  | MM'24 | **90.0 / 67.4** | 83.3 / 60.7 | 68.2 / 47.8 |
| TBSI  | CVPR'23 | 87.1 / 63.7 | 80.6 / 59.9 | 74.9 / 52.3 |
| CAFormer  | AAAI'25 | 88.3 / 66.4 | 81.6 / 60.2 | 74.8 / 53.1 |
| **GAP-Track (Ours)** | ECCV'26 | 87.4 / 64.8 | **84.2 / 62.5** | **75.5 / 54.5** |

### LasHeR

Bold numbers indicate the best performance in each category.

| Algorithm | Source | Full Resolution (PR/SR/NPR) | Half Res 1/2 (PR/SR/NPR) | Quart. Res 1/4 (PR/SR/NPR) |
|-----------|--------|----------------------------|--------------------------|----------------------------|
| MPLT  | arXiv'23 | 72.0 / 57.1 / 68.0 | 65.8 / 52.4 / 61.9 | 55.7 / 43.5 / 51.7 |
| CKD  | MM'24 | **73.2 / 58.1 / 69.3** | 65.7 / 51.5 / 61.6 | 44.1 / 35.3 / 39.4 |
| TBSI  | CVPR'23 | 69.2 / 55.6 / 65.7 | 65.9 / 52.4 / 62.2 | 54.4 / 41.8 / 49.4 |
| CAFormer  | AAAI'25 | 70.0 / 55.6 / 66.1 | 64.3 / 51.2 / 60.6 | 54.8 / 43.6 / 51.1 |
| **GAP-Track (Ours)** | ECCV'26 | 71.8 / 57.1 / 67.9 | **67.6 / 52.9 / 63.3** | **57.2 / 43.7 / 52.7** |


> **Key finding**: GAP-Track delivers superior precision at **1/2 resolution** and continues to significantly outperform the baseline even at **1/4 resolution**, ensuring a robust balance between tracking accuracy and inference efficiency.

## 🚀 Installation

```bash
git clone https://github.com/shijiahahaha/GAPTrack.git
cd GAPTrack

conda create -n gaptrack python=3.8
conda activate gaptrack
pip install -r requirement.txt# GAPTrack
