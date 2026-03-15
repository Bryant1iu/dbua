"""
main.py — DBUA 主程序：基于可微波束合成的超声像差校正

DBUA (Differentiable Beamforming for Ultrasound Aberration correction) 通过
梯度下降优化声速图，使超声图像的聚焦质量指标最优化。

工作流程：
  1. 加载超声 IQ 数据和换能器参数
  2. 构建成像网格和声速优化网格
  3. 扫描全局声速，选择最优初始值
  4. 使用 AMSGrad 优化器迭代优化声速图
  5. 实时可视化 B-mode 图像和声速图，并录制视频

用法：
  python main.py

依赖：
  - PyTorch（自动微分）
  - hdf5storage（读取 .mat 数据文件）
  - matplotlib + ffmpeg（可视化与视频录制）
  - tqdm（进度条）
"""

from pathlib import Path
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from hdf5storage import loadmat
from tqdm import tqdm

from helper import (
    # 核心算法
    das,
    time_of_flight,
    # 损失函数
    lag_one_coherence,
    coherence_factor,
    speckle_brightness,
    phase_error,
    total_variation,
    # 可视化
    imagesc,
    plot_loss_curves,
)


# ==============================================================================
# 全局配置常量
# ==============================================================================

# ---- 优化参数 ----
N_ITERS = 301             # 梯度下降迭代次数
LEARNING_RATE = 10        # AMSGrad 学习率
ASSUMED_C = 1540          # 假设声速 (m/s)，用于计算波长

# ---- B-mode 成像区域 (m) ----
BMODE_X_MIN = -12e-3      # x 方向最小值（左边界）
BMODE_X_MAX = 12e-3       # x 方向最大值（右边界）
BMODE_Z_MIN = 0e-3        # z 方向最小值（浅层）
BMODE_Z_MAX = 40e-3       # z 方向最大值（深层）

# ---- 声速优化网格 (m) ----
SOUND_SPEED_X_MIN = -12e-3
SOUND_SPEED_X_MAX = 12e-3
SOUND_SPEED_Z_MIN = 0e-3
SOUND_SPEED_Z_MAX = 40e-3
SOUND_SPEED_NXC = 19      # x 方向网格点数
SOUND_SPEED_NZC = 31      # z 方向网格点数

# ---- 相位误差估计参数 ----
NXK, NZK = 5, 5           # 核（kernel）大小，每个核约 2λ × 2λ
NXP, NZP = 17, 17         # patch 中心网格大小
PHASE_ERROR_X_MIN = -20e-3
PHASE_ERROR_X_MAX = 20e-3
PHASE_ERROR_Z_MIN = 4e-3
PHASE_ERROR_Z_MAX = 44e-3

# ---- 损失函数选择 ----
# 可选值：
#   "pe" — 相位误差（Phase Error），基于平移孔径相关的相位估计
#   "sb" — 散斑亮度（Speckle Brightness），DAS 输出幅值的均值
#   "cf" — 相干因子（Coherence Factor），相干/非相干叠加比
#   "lc" — 滞后一相干性（Lag-One Coherence），相邻阵元互相关
LOSS = "pe"

# ---- 仿真数据选择 ----
# 均匀声速体模：
#   "1420", "1465", "1480", "1510", "1540", "1555", "1570"
# 非均匀声速体模：
#   "inclusion"       — 含包裹体
#   "inclusion_layer" — 含包裹体和分层
#   "four_layer"      — 四层结构
#   "two_layer"       — 两层结构
#   "checker2"        — 2mm 棋盘格
#   "checker8"        — 8mm 棋盘格
SAMPLE = "checker2"

# 各样本的真实声速 (m/s)，0 表示非均匀体模（无单一真实值）
CTRUE = {
    "1420": 1420, "1465": 1465, "1480": 1480, "1510": 1510,
    "1540": 1540, "1555": 1555, "1570": 1570,
    "inclusion": 0, "inclusion_layer": 0, "four_layer": 0,
    "two_layer": 0, "checker2": 0, "checker8": 0,
}

# 数据文件所在目录
DATA_DIR = Path("./data")


# ==============================================================================
# 数据加载
# ==============================================================================


def load_dataset(sample):
    """从 .mat 文件加载超声仿真数据。

    数据文件包含经过平面波重聚焦的 IQ 数据及相关参数。

    Args:
        sample: 样本名称字符串，对应 DATA_DIR 中的 .mat 文件

    Returns:
        iqdata: IQ 数据数组，形状 [nrx, ntx, n_samples]（numpy 复数数组）
        t0:     各发射阵元的时间零点，形状 [ntx, 1]
        fs:     采样频率 (Hz)
        fd:     解调频率 (Hz)
        elpos:  阵元位置，形状 [3, n_elements]（x, y, z 坐标）
        dsf:    降采样因子
        t:      时间向量
    """
    mdict = loadmat(f"{DATA_DIR}/{sample}.mat")
    iqdata = mdict["iqdata"]
    fs = mdict["fs"][0, 0]        # 采样频率
    fd = mdict["fd"][0, 0]        # 解调频率
    dsf = mdict["dsf"][0, 0]      # 降采样因子
    t = mdict["t"]                # 时间向量
    t0 = mdict["t0"]              # 发射时间零点
    elpos = mdict["elpos"]        # 阵元位置 [x; y; z]
    return iqdata, t0, fs, fd, elpos, dsf, t


# ==============================================================================
# 主程序
# ==============================================================================


def main(sample, loss_name):
    """DBUA 主函数：加载数据 → 初始化 → 优化声速图 → 可视化。

    Args:
        sample:    样本名称（见 CTRUE 字典）
        loss_name: 损失函数名称（"pe"/"sb"/"cf"/"lc"）

    Returns:
        优化后的声速图，形状 [SOUND_SPEED_NXC, SOUND_SPEED_NZC]
    """
    # ---- 参数验证 ----
    assert sample in CTRUE, (
        f'数据样本 "{sample}" 不存在。'
        f'\n可选项: {", ".join(CTRUE.keys())}'
    )

    # ---- 1. 加载并转换数据 ----
    iqdata_np, t0_np, fs, fd, elpos, _, _ = load_dataset(sample)

    # 将 numpy 数据转换为 PyTorch 张量
    # 使用 float64/complex128 保证数值精度（超声信号处理对精度敏感）
    iqdata = torch.tensor(iqdata_np, dtype=torch.complex128)
    t0 = torch.tensor(t0_np, dtype=torch.float64)
    elpos_t = torch.tensor(elpos, dtype=torch.float64)
    xe, _, ze = elpos_t  # 提取阵元的 x 和 z 坐标（忽略 y，假设 2D 成像）

    # 计算波长 λ = c / f（用于确定成像分辨率）
    wavelength = ASSUMED_C / fd

    # ---- 2. 构建 B-mode 成像网格 ----
    # 像素间距 = λ/3（满足 Nyquist 采样定理，保证成像质量）
    xi_1d = torch.arange(BMODE_X_MIN, BMODE_X_MAX, wavelength / 3, dtype=torch.float64)
    zi_1d = torch.arange(BMODE_Z_MIN, BMODE_Z_MAX, wavelength / 3, dtype=torch.float64)
    nxi, nzi = xi_1d.size(0), zi_1d.size(0)
    # 创建 2D 网格（ij 索引：第一维 x，第二维 z）
    xi_grid, zi_grid = np.meshgrid(xi_1d.numpy(), zi_1d.numpy(), indexing="ij")
    xi = torch.tensor(xi_grid, dtype=torch.float64)
    zi = torch.tensor(zi_grid, dtype=torch.float64)

    # ---- 3. 构建声速优化网格 ----
    xc = torch.linspace(SOUND_SPEED_X_MIN, SOUND_SPEED_X_MAX,
                        SOUND_SPEED_NXC, dtype=torch.float64)
    zc = torch.linspace(SOUND_SPEED_Z_MIN, SOUND_SPEED_Z_MAX,
                        SOUND_SPEED_NZC, dtype=torch.float64)
    dxc = xc[1] - xc[0]  # 声速网格 x 间距
    dzc = zc[1] - zc[0]  # 声速网格 z 间距

    # ---- 4. 构建相位误差估计的 patch 网格 ----
    # 核网格：每个 patch 内部的子像素偏移（约 2λ × 2λ 大小）
    xk, zk = np.meshgrid(
        (torch.arange(NXK, dtype=torch.float64).numpy() - (NXK - 1) / 2) * wavelength / 2,
        (torch.arange(NZK, dtype=torch.float64).numpy() - (NZK - 1) / 2) * wavelength / 2,
        indexing="ij",
    )
    # patch 中心网格：在成像视场中均匀分布
    xpc, zpc = np.meshgrid(
        np.linspace(PHASE_ERROR_X_MIN, PHASE_ERROR_X_MAX, NXP),
        np.linspace(PHASE_ERROR_Z_MIN, PHASE_ERROR_Z_MAX, NZP),
        indexing="ij",
    )

    # ---- 5. 维度广播准备 ----
    # 目标维度：[n_elements, n_patch_centers, n_kernel_pixels]
    xe = xe.reshape(-1, 1, 1)  # 阵元坐标 → [n_elem, 1, 1]
    ze = ze.reshape(-1, 1, 1)
    xp = (torch.tensor(xpc, dtype=torch.float64).reshape(1, -1, 1)
          + torch.tensor(xk, dtype=torch.float64).reshape(1, 1, -1))
    zp = (torch.tensor(zpc, dtype=torch.float64).reshape(1, -1, 1)
          + torch.tensor(zk, dtype=torch.float64).reshape(1, 1, -1))
    # 手动广播，确保 xp 和 zp 形状完全一致
    xp = xp + 0 * zp
    zp = zp + 0 * xp

    # ---- 6. 定义飞行时间与成像函数 ----

    def compute_tof_image(c):
        """计算成像像素到阵元的飞行时间。"""
        return time_of_flight(xe, ze, xi, zi, xc, zc, c, fnum=0.5, npts=64)

    def compute_tof_patch(c):
        """计算 patch 像素到阵元的飞行时间。"""
        return time_of_flight(xe, ze, xp, zp, xc, zc, c, fnum=0.5, npts=64)

    def make_bmode_image(c):
        """根据声速图生成 B-mode 图像（取幅值）。"""
        t = compute_tof_image(c)
        return torch.abs(das(iqdata, t - t0, t, fs, fd))

    def apply_loss_metric(metric_func, c):
        """对给定的图像质量指标函数，计算其在 patch 网格上的值。"""
        t = compute_tof_patch(c)
        return metric_func(iqdata, t - t0, t, fs, fd)

    # ---- 7. 定义损失函数 ----

    def sb_loss(c):
        """散斑亮度损失：1 - SB（最小化 → 最大化散斑亮度）。"""
        return 1 - apply_loss_metric(speckle_brightness, c)

    def lc_loss(c):
        """滞后一相干性损失：1 - mean(LOC)。"""
        return 1 - torch.mean(apply_loss_metric(lag_one_coherence, c))

    def cf_loss(c):
        """相干因子损失：1 - mean(CF)。"""
        return 1 - torch.mean(apply_loss_metric(coherence_factor, c))

    def pe_loss(c):
        """相位误差损失：log(1 + (100·Δφ)²) 的均值。

        使用 log1p 使损失对大相位误差不那么敏感（鲁棒损失）。
        乘以 100 是为了在小相位误差时仍有足够的梯度。
        """
        t = compute_tof_patch(c)
        dphi = phase_error(iqdata, t - t0, t, fs, fd)
        # 将零值标记为 NaN，避免影响均值计算
        valid = dphi != 0
        dphi = torch.where(valid, dphi, torch.tensor(float('nan'), dtype=dphi.dtype))
        return torch.nanmean(torch.log1p(torch.square(100 * dphi)))

    def tv_regularization(c):
        """全变分正则化项（乘以网格间距进行物理量纲归一化）。"""
        return total_variation(c) * dxc * dzc

    # 损失函数分发器
    loss_functions = {
        "sb": sb_loss,
        "lc": lc_loss,
        "cf": cf_loss,
        "pe": pe_loss,
    }

    def compute_loss(c):
        """计算总损失 = 聚焦质量损失 + TV 正则化。"""
        if loss_name not in loss_functions:
            raise NotImplementedError(f"未知损失函数: {loss_name}")
        return loss_functions[loss_name](c) + tv_regularization(c) * 1e2

    # ---- 8. 全局声速扫描：寻找最优初始值 ----
    print("扫描全局声速以确定最优初始值...")
    c0_range = np.linspace(1340, 1740, 201)

    def eval_loss_at_speed(loss_fn, speed):
        """在给定全局声速下计算损失值。"""
        c_uniform = speed * torch.ones(
            (SOUND_SPEED_NXC, SOUND_SPEED_NZC), dtype=torch.float64
        )
        return loss_fn(c_uniform).item()

    dsb = np.array([eval_loss_at_speed(sb_loss, cc) for cc in c0_range])
    dlc = np.array([eval_loss_at_speed(lc_loss, cc) for cc in c0_range])
    dcf = np.array([eval_loss_at_speed(cf_loss, cc) for cc in c0_range])
    dpe = np.array([eval_loss_at_speed(pe_loss, cc) for cc in c0_range])

    # 选择使相位误差最小的声速作为初始值
    optimal_c0 = c0_range[np.argmin(dpe)]
    print(f"最优初始声速: {optimal_c0:.1f} m/s")

    # 绘制损失曲线
    plot_loss_curves(c0_range, dsb, dlc, dcf, dpe, sample)

    # ---- 9. 初始化优化变量和优化器 ----
    # 声速图作为可优化参数（requires_grad=True）
    c = torch.nn.Parameter(
        optimal_c0 * torch.ones(
            (SOUND_SPEED_NXC, SOUND_SPEED_NZC), dtype=torch.float64
        )
    )
    # AMSGrad 优化器（Adam 的变体，收敛性更好）
    optimizer = torch.optim.Adam([c], lr=LEARNING_RATE, amsgrad=True)

    # ---- 10. 设置可视化 ----
    fig, _ = plt.subplots(1, 2, figsize=[9, 4])
    vobj = FFMpegWriter(fps=30)
    vobj.setup(fig, "videos/%s_opt%s.mp4" % (sample, loss_name), dpi=144)

    # 坐标轴范围（毫米单位，便于显示）
    ximm = xi[:, 0].numpy() * 1e3  # x 坐标 (mm)
    zimm = zi[0, :].numpy() * 1e3  # z 坐标 (mm)
    xcmm = xc.numpy() * 1e3
    zcmm = zc.numpy() * 1e3

    # 显示参数
    bmode_dynamic_range = [-45, +5]  # B-mode 动态范围 (dB)
    # 声速图颜色范围
    c_true = CTRUE[sample]
    if c_true > 0:
        # 均匀体模：以真实值为中心，±50 m/s
        c_display_range = np.array([-50, +50]) + c_true
        c_colormap = "seismic"  # 红蓝色图，便于看偏差
    else:
        # 非均匀体模：固定范围
        c_display_range = [1400, 1600]
        c_colormap = "jet"

    # ---- 11. 定义绘图函数 ----

    def update_figure(c_param, iteration, handles=None):
        """创建或更新 B-mode 图像和声速图的双面板显示。

        第一次调用（handles=None）时创建图形，后续调用更新数据。

        Args:
            c_param:    当前声速图参数
            iteration:  当前迭代次数
            handles:    之前创建的图形句柄元组，None 表示首次创建

        Returns:
            图形句柄元组 (bmode_image, cmap_image, bmode_title, cmap_title, bmax)
        """
        with torch.no_grad():
            c_detached = c_param.detach()
            bmode = make_bmode_image(c_detached)

        # 首次调用时确定归一化基准
        if handles is None:
            bmax = bmode.max().item()
        else:
            hbi, hci, hbt, hct, bmax = handles

        # 转换为对数刻度 (dB)
        bmode_normalized = bmode / bmax
        bmode_normalized = bmode_normalized + 1e-10 * (bmode_normalized == 0)  # 避免 log(0)
        bmode_db = 20 * torch.log10(bmode_normalized)
        bmode_db = bmode_db.reshape(nxi, nzi).T.numpy()

        # 声速图转为 numpy
        c_display = c_detached.reshape(SOUND_SPEED_NXC, SOUND_SPEED_NZC).T.numpy()

        if handles is None:
            # ---- 首次创建图形 ----
            # 性能基准测试
            tic = time.perf_counter_ns()
            with torch.no_grad():
                for _ in range(30):
                    make_bmode_image(c_detached)
            toc = time.perf_counter_ns()
            fps = 100.0 / ((toc - tic) * 1e-9)
            print(f"PyTorch 波束合成速度: {fps:.1f} fps")

            fig.clf()

            # 左面板：B-mode 图像
            plt.subplot(121)
            hbi = imagesc(ximm, zimm, bmode_db, bmode_dynamic_range,
                          cmap="bone", interpolation="bicubic")
            with torch.no_grad():
                hbt = plt.title(
                    "SB: %.2f, CF: %.3f, PE: %.3f" % (
                        sb_loss(c_detached).item(),
                        cf_loss(c_detached).item(),
                        pe_loss(c_detached).item(),
                    )
                )
            plt.xlim(ximm[0], ximm[-1])
            plt.ylim(zimm[-1], zimm[0])

            # 右面板：声速图
            plt.subplot(122)
            hci = imagesc(xcmm, zcmm, c_display, c_display_range,
                          cmap=c_colormap, interpolation="bicubic")
            if c_true > 0:
                hct = plt.title(
                    "Iteration %d: MAE %.2f" % (iteration, np.mean(np.abs(c_display - c_true)))
                )
            else:
                hct = plt.title(
                    "Iteration %d: Mean value %.2f" % (iteration, np.mean(c_display))
                )
            plt.xlim(ximm[0], ximm[-1])
            plt.ylim(zimm[-1], zimm[0])

            fig.tight_layout()
            return hbi, hci, hbt, hct, bmax

        else:
            # ---- 更新已有图形 ----
            hbi.set_data(bmode_db)
            hci.set_data(c_display)
            with torch.no_grad():
                hbt.set_text(
                    "SB: %.2f, CF: %.3f, PE: %.3f" % (
                        sb_loss(c_detached).item(),
                        cf_loss(c_detached).item(),
                        pe_loss(c_detached).item(),
                    )
                )
            if c_true > 0:
                hct.set_text(
                    "Iteration %d: MAE %.2f" % (iteration, np.mean(np.abs(c_display - c_true)))
                )
            else:
                hct.set_text(
                    "Iteration %d: Mean value %.2f" % (iteration, np.mean(c_display))
                )

        plt.savefig(f"scratch/{sample}.png")

    # ---- 12. 初始化图形 ----
    handles = update_figure(c, 0)

    # ---- 13. 优化循环 ----
    print(f"开始优化（{N_ITERS} 次迭代，损失函数: {loss_name}）...")
    for i in tqdm(range(N_ITERS)):
        # PyTorch 标准优化步骤：清零梯度 → 前向计算 → 反向传播 → 参数更新
        optimizer.zero_grad()
        loss_val = compute_loss(c)
        loss_val.backward()
        optimizer.step()

        # 更新可视化并录制视频帧
        update_figure(c, i + 1, handles)
        vobj.grab_frame()

    # 关闭视频写入器
    vobj.finish()
    print(f"优化完成。视频保存至 videos/{sample}_opt{loss_name}.mp4")

    return c.detach()


# ==============================================================================
# 入口点
# ==============================================================================

if __name__ == "__main__":
    main(SAMPLE, LOSS)

    # # 批量运行所有样本：
    # for sample in CTRUE.keys():
    #     print(f"处理样本: {sample}")
    #     main(sample, LOSS)
