# Differentiable Beamforming for Ultrasound Autofocusing (DBUA)

### [Project Page](https://www.waltersimson.com/dbua) | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_41) | [Pre-Print](https://waltersimson.com/dbua/static/pdfs/SimsonMICCAI2023.pdf) | [Data](https://github.com/waltsims/dbua/releases/tag/miccai2023)

[Walter Simson](https://waltersimson.com/),
[Louise Zhuang](https://profiles.stanford.edu/louise-zhuang),
[Sergio Sanabria](https://scholar.google.es/citations?hl=es&user=E7h77bAAAAAJ),
[Neha Antil](https://med.stanford.edu/profiles/neha-antil),
[Jeremy Dahl](https://med.stanford.edu/profiles/jeremy-dahl),
[Dongwoon Hyun](https://profiles.stanford.edu/dongwoon-hyun)<br>
Stanford University

This is the official implementation of "Differentiable Beamforming for Ultrasound Autofocusing", refactored to use **PyTorch** (replacing JAX).

[![dbua_video](https://img.youtube.com/vi/cUoAsEA5snE/0.jpg)](https://www.youtube.com/watch?v=cUoAsEA5snE)

## Overview

DBUA 通过可微波束合成（differentiable beamforming）和梯度下降优化声速图（speed-of-sound map），实现超声图像的自动像差校正（aberration correction）。

**工作流程：**

1. 加载超声 IQ 数据和换能器参数
2. 构建成像网格和声速优化网格
3. 扫描全局声速，选择最优初始值
4. 使用 AMSGrad 优化器迭代优化声速图
5. 实时可视化 B-mode 图像和声速图，并录制视频

## 项目结构

| 文件 | 说明 |
|------|------|
| `main.py` | 主程序入口，包含数据加载、优化循环、可视化和全局配置 |
| `helper.py` | 工具模块，包含插值、DAS 波束合成、飞行时间计算、损失函数和绘图工具 |

### helper.py 模块组成

- **插值函数** — 最近邻、线性、三次 Hermite (Catmull-Rom)、Lanczos 插值
- **延时叠加 (DAS)** — 超声波束合成核心算法，支持 `torch.vmap` 向量化和 gradient checkpointing
- **飞行时间 (TOF)** — 基于声速图的直线路径慢度积分，支持 F-number 孔径约束
- **损失函数** — 散斑亮度 (SB)、滞后一相干性 (LOC)、相干因子 (CF)、相位误差 (PE)、全变分 (TV) 正则化
- **可视化工具** — MATLAB 风格的 `imagesc`、损失曲线绘制

## 依赖

- Python 3.x
- **PyTorch**（自动微分与张量计算）
- hdf5storage（读取 `.mat` 数据文件）
- matplotlib + ffmpeg（可视化与视频录制）
- tqdm（进度条）
- NumPy

## 安装

1. 安装 ffmpeg（用于保存 mp4 视频）：

```bash
sudo apt-get install ffmpeg
```

2. 安装 Python 依赖：

```bash
pip install torch numpy hdf5storage matplotlib tqdm
```

或使用 conda/micromamba：

```bash
micromamba env create -f environment.yml
micromamba activate dbua
```

## 数据准备

从 [GitHub Releases](https://github.com/waltsims/dbua/releases/tag/miccai2023) 下载数据文件，将 `.mat` 文件放入 `data/` 目录：

```
data/
├── 1420.mat          # 均匀声速体模 (1420 m/s)
├── 1465.mat
├── 1480.mat
├── 1510.mat
├── 1540.mat
├── 1555.mat
├── 1570.mat
├── checker2.mat      # 2mm 棋盘格体模
├── checker8.mat      # 8mm 棋盘格体模
├── four_layer.mat    # 四层结构体模
├── inclusion.mat     # 含包裹体体模
├── inclusion_layer.mat
└── two_layer.mat     # 两层结构体模
```

## 使用方法

运行默认配置：

```bash
python main.py
```

### 配置参数

在 `main.py` 顶部可修改以下全局配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_ITERS` | 301 | 梯度下降迭代次数 |
| `LEARNING_RATE` | 10 | AMSGrad 学习率 |
| `LOSS` | `"pe"` | 损失函数：`"pe"` 相位误差 / `"sb"` 散斑亮度 / `"cf"` 相干因子 / `"lc"` 滞后一相干性 |
| `SAMPLE` | `"checker2"` | 数据样本名称 |

### 输出

- `videos/<sample>_opt<loss>.mp4` — 优化过程视频
- `images/losses_<sample>.png` — 各损失函数随声速变化曲线
- `scratch/<sample>.png` — 最终 B-mode 图像和声速图

## Citation

```bibtex
@inproceedings{simson2023dbua,
    title={Differentiable Beamforming for Ultrasound Autofocusing},
    author={Simson, Walter and Zhuang, Louise and Sanabria, Sergio J and Antil, Neha and Dahl, Jeremy J and Hyun, Dongwoon},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    pages={428--437},
    year={2023},
    organization={Springer}
}
```

## FAQ

- **Q:** 原始版本使用 JAX，为什么改为 PyTorch？
- **A:** PyTorch 生态更成熟，安装配置更简单，且通过 `torch.vmap` 和 gradient checkpointing 可实现类似的向量化和内存优化。

- **Q:** 推荐的运行配置？
- **A:**

| 属性 | 推荐配置 |
|------|----------|
| OS | Ubuntu Linux |
| RAM | 32GB+ |
| GPU | NVIDIA GPU (48 GB VRAM) |
| CUDA | 12.1+ |
