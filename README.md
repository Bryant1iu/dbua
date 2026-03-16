# DBUA：用于超声自动聚焦的可微分波束形成（PyTorch 版本）

本仓库实现了 **Differentiable Beamforming for Ultrasound Autofocusing** 的完整实验流程：
通过可微分的延时叠加（DAS）成像链路，把图像聚焦质量指标作为优化目标，迭代求解声速图（speed-of-sound map）。

> 论文：Differentiable Beamforming for Ultrasound Autofocusing（MICCAI 2023）

- Project page: https://www.waltersimson.com/dbua
- Paper: https://link.springer.com/chapter/10.1007/978-3-031-43999-5_41
- Preprint: https://waltersimson.com/dbua/static/pdfs/SimsonMICCAI2023.pdf
- Data release: https://github.com/waltsims/dbua/releases/tag/miccai2023

---

## 项目当前代码结构（以本仓库实际文件为准）

```text
.
├── main.py              # 主程序：数据加载、初始化、损失定义、优化、可视化、视频写出
├── helper.py            # 核心算法库：插值、DAS、TOF、损失指标、绘图工具
├── environment.yml      # Conda 环境依赖
├── data/                # 放置 .mat 数据文件
├── images/              # 保存 loss 曲线图
├── videos/              # 保存优化过程视频
└── scratch/             # 保存中间可视化图
```

> 说明：当前可执行入口是 `main.py`（不是 `dbua.py`）。

---

## 方法概览

`main.py` 的主流程如下：

1. 从 `data/<sample>.mat` 读取 IQ 数据与探头参数。
2. 构建 B-mode 成像网格与声速优化网格。
3. 基于 `time_of_flight` + `das` 构建可微成像链路。
4. 定义聚焦质量损失（SB / LOC / CF / PE）以及 TV 正则。
5. 先扫描全局声速（1340~1740 m/s）选择较优初始化。
6. 使用 Adam(AMSGrad) 迭代优化声速图。
7. 每轮刷新 B-mode 与声速图，可输出 MP4 视频与中间 PNG。

---

## 环境准备

### 1) 安装 ffmpeg（用于写出 mp4）

Ubuntu 示例：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

### 2) 创建 Python 环境

```bash
micromamba env create -f environment.yml
micromamba activate dbua
```

`environment.yml` 中主要依赖包括：

- python 3.9
- hdf5storage
- matplotlib
- numpy
- tqdm
- jaxopt / optax（环境中保留）

> 另外请确保环境中可用 `torch`（本实现核心依赖 PyTorch 自动微分）。

---

## 数据准备

从 release 下载数据后，把 `.mat` 文件放到 `data/` 目录。例如：

```text
data
├── 1420.mat
├── 1465.mat
├── 1480.mat
├── 1510.mat
├── 1540.mat
├── 1555.mat
├── 1570.mat
├── checker2.mat
├── checker8.mat
├── four_layer.mat
├── inclusion.mat
├── inclusion_layer.mat
└── two_layer.mat
```

---

## 运行

```bash
python main.py
```

程序会：

- 在 `images/` 生成全局声速扫描损失曲线（`losses_<sample>.png`）
- 在 `videos/` 生成优化过程视频（`<sample>_opt<loss>.mp4`）
- 在 `scratch/` 写出中间可视化图片

---

## 可配置项（`main.py` 顶部常量）

你可以在 `main.py` 中修改以下参数：

- `SAMPLE`：选择样本（如 `checker2`、`inclusion`、`1540` 等）
- `LOSS`：优化目标（`"pe"`, `"sb"`, `"cf"`, `"lc"`）
- `N_ITERS`：优化迭代次数
- `LEARNING_RATE`：优化学习率
- 成像区域与声速网格参数（`BMODE_*`, `SOUND_SPEED_*`）
- phase error patch 参数（`NXK/NZK/NXP/NZP` 等）

---

## `helper.py` 主要模块

### 1) 插值（Interpolation）

- `interp_nearest`
- `interp_linear`
- `interp_cubic`（默认）
- `interp_lanczos`（`lanczos3/lanczos5`）

### 2) 可微延时叠加 DAS

- `das(...)` 支持
  - 发射/接收维线性组合矩阵 `A/B`
  - 孔径加权 `apoA/apoB`
  - 多种插值策略

### 3) 飞行时间计算 TOF

- `time_of_flight(...)`
  - 在声速图上进行慢度积分（直线路径近似）
  - 含 F-number 和最小孔径约束

### 4) 图像质量指标 / 损失相关函数

- `speckle_brightness`
- `lag_one_coherence`
- `coherence_factor`
- `phase_error`
- `total_variation`

### 5) 可视化工具

- `imagesc`
- `plot_loss_curves`

---

## 结果与输出说明

- **均匀体模**：可通过真实声速 `CTRUE` 计算 MAE（显示在图标题）。
- **非均匀体模**：显示优化中声速图的均值。
- B-mode 以 dB 显示，默认动态范围约 `[-45, +5]`。

---

## 常见问题

### 1) 运行时报找不到数据文件
请确认 `data/<sample>.mat` 存在，且 `SAMPLE` 名称与文件名一致。

### 2) 不能写出视频
通常是系统缺少 ffmpeg，先安装后重试。

### 3) 显存/内存压力较大
可尝试：

- 减小 `N_ITERS`
- 降低网格分辨率（如减少 `SOUND_SPEED_NXC/NZC`）
- 缩小 patch 尺寸（`NXK/NZK/NXP/NZP`）

---

## 引用

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

---

## License

MIT License. See `LICENSE` for details.
