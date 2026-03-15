"""
helper.py — DBUA (Differentiable Beamforming for Ultrasound Aberration correction) 工具模块

本模块包含超声波束合成与声速优化所需的全部辅助函数，分为以下几个部分：
  1. 插值函数（Interpolation）      —— 1D 信号插值方法
  2. 延时叠加（Delay-and-Sum, DAS）  —— 超声波束合成核心算法
  3. 飞行时间（Time-of-Flight）      —— 基于声速图的飞行时间计算
  4. 损失函数（Loss Functions）      —— 图像质量评估与优化目标
  5. 可视化工具（Visualization）     —— B-mode 成像与声速图绘制

技术栈：PyTorch（支持自动微分，用于声速图优化）
"""

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import matplotlib.pyplot as plt


# ==============================================================================
# 第一部分：1D 信号插值函数
# ==============================================================================
# 超声波束合成需要在非整数采样点处获取 IQ 信号值，因此需要各种插值方法。
# 所有插值函数的接口统一为 f(x, si)，其中 x 是 1D 信号，si 是浮点索引。


def safe_access(signal, indices):
    """安全数组访问：对越界索引返回 0，避免数组越界错误。

    在插值过程中，浮点索引的相邻整数点可能落在信号边界之外。
    此函数将越界索引钳位（clamp）到有效范围内，但对越界位置返回零值，
    确保插值结果在边界处平滑衰减到零。

    Args:
        signal: 一维信号张量，形状 [n_samples]
        indices: 浮点或整数索引张量，形状任意（支持广播）

    Returns:
        与 indices 同形状的张量，越界位置填充为 0
    """
    indices = indices.to(torch.int64)
    # 标记哪些索引在有效范围 [0, n_samples) 内
    valid_mask = (indices >= 0) & (indices < signal.numel())
    # 将越界索引钳位到有效范围，防止索引错误
    clamped_indices = torch.clamp(indices, 0, signal.numel() - 1)
    # 有效位置取真实值，无效位置取零
    return torch.where(valid_mask, signal[clamped_indices],
                       torch.zeros_like(signal[clamped_indices]))


def interp_nearest(signal, sample_indices):
    """最近邻插值（Nearest Neighbor Interpolation）。

    最简单的插值方法：取最近的整数采样点的值。
    速度最快，但会产生阶梯状伪影。

    Args:
        signal: 一维信号张量，形状 [n_samples]
        sample_indices: 浮点采样索引，形状任意

    Returns:
        插值后的信号值
    """
    rounded = torch.clamp(torch.round(sample_indices), 0, signal.shape[0] - 1)
    return signal[rounded.to(torch.int64)]


def interp_linear(signal, sample_indices):
    """线性插值（Linear Interpolation）。

    在两个相邻采样点之间进行线性内插。
    计算量小，适用于对精度要求不高的场景。

    数学公式：y = (1-f)*x[s] + f*x[s+1]
    其中 s = floor(si), f = si - s（小数部分）

    Args:
        signal: 一维信号张量，形状 [n_samples]
        sample_indices: 浮点采样索引，形状任意

    Returns:
        插值后的信号值
    """
    frac = sample_indices - torch.floor(sample_indices)  # 小数部分
    base = torch.floor(sample_indices)                    # 整数部分
    x0 = safe_access(signal, base + 0)
    x1 = safe_access(signal, base + 1)
    return (1 - frac) * x0 + frac * x1


def interp_cubic(signal, sample_indices):
    """三次 Hermite 插值（Cubic Hermite Interpolation）。

    使用 4 个相邻采样点进行三次多项式插值，在精度与计算量之间取得良好平衡。
    这是波束合成中最常用的插值方法（默认方法）。

    使用 Catmull-Rom 样条系数：
      a0 = f*(-1 + f*(2f - 1))          / 2
      a1 = (2 + f*(0 + f*(-5f + 3)))    / 2
      a2 = f*(1 + f*(4f - 3))           / 2
      a3 = f*(0 + f*(-f + 1))           / 2

    Args:
        signal: 一维信号张量，形状 [n_samples]
        sample_indices: 浮点采样索引，形状任意

    Returns:
        插值后的信号值
    """
    frac = sample_indices - torch.floor(sample_indices)  # 小数部分
    base = torch.floor(sample_indices)                    # 整数部分
    # 取 4 个相邻采样点的值
    x0 = safe_access(signal, base - 1)
    x1 = safe_access(signal, base + 0)
    x2 = safe_access(signal, base + 1)
    x3 = safe_access(signal, base + 2)
    # Catmull-Rom 样条的多项式系数（Horner 形式，减少乘法次数）
    a0 = 0 + frac * (-1 + frac * (+2 * frac - 1))
    a1 = 2 + frac * (+0 + frac * (-5 * frac + 3))
    a2 = 0 + frac * (+1 + frac * (+4 * frac - 3))
    a3 = 0 + frac * (+0 + frac * (-1 * frac + 1))
    return (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3) / 2


def _lanczos_kernel(x, nlobe=3):
    """Lanczos 插值核函数。

    Lanczos 核 = sinc(x) * sinc(x/a)，其中 a = (nlobe+1)/2。
    在 |x| >= a 时为零（有限支撑）。

    Args:
        x: 输入张量
        nlobe: sinc 函数的瓣数（3 或 5）

    Returns:
        核函数值
    """
    a = (nlobe + 1) / 2
    # torch.sinc 使用归一化 sinc: sinc(x) = sin(pi*x)/(pi*x)
    return torch.where(
        torch.abs(x) < a,
        torch.sinc(x) * torch.sinc(x / a),
        torch.zeros_like(x)
    )


def interp_lanczos(signal, sample_indices, nlobe=3):
    """Lanczos 插值（Lanczos Interpolation）。

    基于 sinc 函数的高质量插值方法，比三次插值更精确，但计算量更大。
    nlobe=3 (Lanczos-3) 和 nlobe=5 (Lanczos-5) 是常用配置。

    Args:
        signal: 一维信号张量，形状 [n_samples]
        sample_indices: 浮点采样索引，形状任意
        nlobe: Lanczos 核的瓣数（默认 3）

    Returns:
        插值后的信号值
    """
    frac = sample_indices - torch.floor(sample_indices)
    base = torch.floor(sample_indices)
    # 取 4 个相邻采样点
    x0 = safe_access(signal, base - 1)
    x1 = safe_access(signal, base + 0)
    x2 = safe_access(signal, base + 1)
    x3 = safe_access(signal, base + 2)
    # 计算 Lanczos 核权重
    a0 = _lanczos_kernel(frac + 1, nlobe)
    a1 = _lanczos_kernel(frac + 0, nlobe)
    a2 = _lanczos_kernel(frac - 1, nlobe)
    a3 = _lanczos_kernel(frac - 2, nlobe)
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3


# 插值方法注册表：字符串名称 -> 插值函数的映射
INTERPOLATION_METHODS = {
    "nearest": interp_nearest,
    "linear": interp_linear,
    "cubic": interp_cubic,
    "lanczos3": lambda x, t: interp_lanczos(x, t, nlobe=3),
    "lanczos5": lambda x, t: interp_lanczos(x, t, nlobe=5),
}


# ==============================================================================
# 第二部分：延时叠加波束合成（Delay-and-Sum Beamforming）
# ==============================================================================
# DAS 是超声成像的核心算法：根据声波传播的飞行时间（time-of-flight）对
# 各阵元接收到的 IQ 信号进行延时对齐，然后叠加（或加权组合），形成聚焦图像。


def das(iqraw, tA, tB, fs, fd, A=None, B=None, apoA=1, apoB=1, interp="cubic"):
    """延时叠加（Delay-and-Sum）波束合成。

    根据给定的时间延迟剖面对 IQ 数据进行波束合成。支持任意线性组合方式，
    可用于常规聚焦、子孔径波束合成、延时但不叠加等多种模式。

    算法流程：
      1. 对每个发射-接收阵元对，根据飞行时间 tA+tB 对 IQ 信号进行插值
      2. 乘以基带相位校正因子 exp(2πj·fd·t) 进行解调
      3. 乘以孔径加权（apodization）apoA, apoB
      4. 通过张量缩并（tensordot）按矩阵 A、B 进行线性组合

    内存优化：外层循环逐发射阵元处理，内层使用 vmap 向量化接收阵元，
    并通过 gradient checkpointing 减少内存占用。

    Args:
        iqraw:  原始基带 IQ 数据，形状 [na, nb, n_samples]
                na = 发射阵元数（或第一维度），nb = 接收阵元数（或第二维度）
        tA:     第一维度的时间延迟，形状 [na, *pixel_dims]
        tB:     第二维度的时间延迟，形状 [nb, *pixel_dims]
        fs:     采样频率 (Hz)，用于将时间转换为采样点索引
        fd:     解调频率 (Hz)，0 表示射频调制数据
        A:      第一维度的线性组合矩阵，形状 [*na_out, na]
                None = 全部求和（等效于全 1 向量）
                单位矩阵 = 延时但不叠加
        B:      第二维度的线性组合矩阵，形状 [*nb_out, nb]
        apoA:   第一维度的孔径加权，形状 [na, *pixel_dims]，支持广播
        apoB:   第二维度的孔径加权，形状 [nb, *pixel_dims]，支持广播
        interp: 插值方法名称 ("nearest"/"linear"/"cubic"/"lanczos3"/"lanczos5")

    Returns:
        聚焦后的 IQ 数据，形状 [*na_out, *nb_out, *pixel_dims]

    Note:
        根据声学互易原理（acoustic reciprocity），a 和 b 哪个对应发射、
        哪个对应接收不影响结果。
    """
    # 默认线性组合：对所有阵元求和
    if A is None:
        A = torch.ones((iqraw.shape[0],), dtype=iqraw.dtype, device=iqraw.device)
    if B is None:
        B = torch.ones((iqraw.shape[1],), dtype=iqraw.dtype, device=iqraw.device)

    # 选择插值函数
    fint = INTERPOLATION_METHODS[interp]

    def baseband_interpolate(iq, t):
        """基带插值：对 IQ 信号进行时间延迟插值并校正基带相位。

        Args:
            iq: 单通道 IQ 信号，形状 [n_samples]
            t:  时间延迟，形状 [*pixel_dims]

        Returns:
            聚焦后的 IQ 值，形状 [*pixel_dims]
        """
        # 将时间延迟转换为采样点索引并插值
        iq_focused = fint(iq, fs * t)
        # 乘以基带相位校正因子（解调补偿）
        return iq_focused * torch.exp(2j * torch.pi * fd * t)

    def das_single_transmit(iq_i, tA_i):
        """对单个发射阵元，向量化处理所有接收阵元的波束合成。

        使用 torch.vmap 对接收维度进行向量化，避免显式循环。

        Args:
            iq_i:  单个发射阵元的所有接收通道 IQ 数据，形状 [nb, n_samples]
            tA_i:  单个发射阵元的时间延迟，形状 [*pixel_dims]

        Returns:
            经 B 矩阵线性组合后的结果，形状 [*nb_out, *pixel_dims]
        """
        # vmap 在接收阵元维度（dim 0）上向量化 baseband_interpolate
        results = torch.vmap(baseband_interpolate)(iq_i, tA_i + tB) * apoB
        # 通过张量缩并按 B 矩阵组合接收阵元
        return torch.tensordot(B, results, dims=([-1], [0]))

    # 外层循环遍历发射阵元，使用 gradient checkpointing 节省内存
    # （前向传播时不保存中间结果，反向传播时重新计算）
    results = []
    for i in range(iqraw.shape[0]):
        result = torch_checkpoint(
            das_single_transmit, iqraw[i], tA[i], use_reentrant=False
        )
        results.append(result)
    # 堆叠所有发射阵元的结果，乘以发射孔径加权
    mapped_results = torch.stack(results) * apoA

    # 通过张量缩并按 A 矩阵组合发射阵元
    return torch.tensordot(A, mapped_results, dims=([-1], [0]))


# ==============================================================================
# 第三部分：飞行时间计算（Time-of-Flight）
# ==============================================================================
# 飞行时间（TOF）是声波从一个点传播到另一个点所需的时间。
# 在非均匀声速介质中，TOF 需要沿路径积分慢度（slowness = 1/c）。
# 这里使用直线路径近似（straight ray approximation）。


def time_of_flight(x0, z0, x1, z1, xc, zc, c, fnum, npts, Dmin=3e-3):
    """计算从 (x0,z0) 到 (x1,z1) 的飞行时间。

    基于声速图 c 沿直线路径积分慢度（slowness = 1/c），使用双线性插值
    获取路径上各点的慢度值。同时应用 F-number 孔径约束，屏蔽过大角度的路径。

    算法步骤：
      1. 将直线路径参数化为 t ∈ [0, 1]，取 npts 个采样点
      2. 对每个采样点，双线性插值得到该位置的慢度
      3. 取所有采样点慢度的均值 × 路径长度 = 飞行时间
      4. 根据 F-number 和最小孔径约束，屏蔽无效路径

    Args:
        x0, z0: 路径起点坐标，形状任意（支持广播）
        x1, z1: 路径终点坐标，形状任意（支持广播）
        xc:     声速图的 x 轴网格点，形状 [nxc]
        zc:     声速图的 z 轴网格点，形状 [nzc]
        c:      声速图，形状 [nxc, nzc]，单位 m/s
        fnum:   F 数（f-number），控制有效孔径角度
        npts:   路径上的积分采样点数
        Dmin:   最小孔径宽度 (m)，用于避免极小角度时孔径过窄

    Returns:
        飞行时间张量，无效区域填充为 -10（后续插值会产生零值）
    """
    # 将路径参数化：t=0 为起点，t=1 为终点
    # linspace(1,0,npts).flip(0) 等效于 linspace(0,1,npts) 但不含端点 0
    t_all = torch.linspace(1, 0, npts).flip(0)

    # 慢度图（slowness = 1/c），单位 s/m
    slowness_map = 1 / c

    def bilinear_interpolate_slowness(t):
        """在参数 t 处对慢度图进行双线性插值。

        将路径上参数 t 对应的空间坐标 (xt, zt) 转换为慢度图中的网格索引，
        然后使用双线性插值获取该位置的慢度值。

        Args:
            t: 路径参数，标量，范围 [0, 1]

        Returns:
            插值得到的慢度值，形状与 x0, x1 广播后一致
        """
        # 计算路径上 t 处的真实空间坐标
        xt = t * (x1 - x0) + x0  # x 方向位置
        zt = t * (z1 - z0) + z0  # z 方向位置

        # 将空间坐标转换为慢度图中的连续索引
        dxc = xc[1] - xc[0]  # x 方向网格间距
        dzc = zc[1] - zc[0]  # z 方向网格间距
        # 连续索引，钳位到有效范围 [0, n-1]
        xit = torch.clamp((xt - xc[0]) / dxc, 0, slowness_map.shape[0] - 1)
        zit = torch.clamp((zt - zc[0]) / dzc, 0, slowness_map.shape[1] - 1)

        # 双线性插值的四个角点索引
        xi0 = torch.floor(xit)
        zi0 = torch.floor(zit)
        xi1 = xi0 + 1
        zi1 = zi0 + 1

        # 获取四个角点的慢度值（钳位索引防止越界）
        s00 = slowness_map[xi0.to(torch.int64),
                           zi0.to(torch.int64)]
        s10 = slowness_map[xi1.to(torch.int64).clamp(max=slowness_map.shape[0] - 1),
                           zi0.to(torch.int64)]
        s01 = slowness_map[xi0.to(torch.int64),
                           zi1.to(torch.int64).clamp(max=slowness_map.shape[1] - 1)]
        s11 = slowness_map[xi1.to(torch.int64).clamp(max=slowness_map.shape[0] - 1),
                           zi1.to(torch.int64).clamp(max=slowness_map.shape[1] - 1)]

        # 双线性插值权重
        w00 = (xi1 - xit) * (zi1 - zit)
        w10 = (xit - xi0) * (zi1 - zit)
        w01 = (xi1 - xit) * (zit - zi0)
        w11 = (xit - xi0) * (zit - zi0)

        return s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11

    # 计算路径的几何信息
    dx = torch.abs(x1 - x0)  # x 方向距离
    dz = torch.abs(z1 - z0)  # z 方向距离
    path_length = torch.sqrt(dx**2 + dz**2)  # 欧几里得路径长度

    # 沿路径积分慢度：对所有采样点的慢度取均值
    slowness_samples = torch.stack([bilinear_interpolate_slowness(t) for t in t_all])
    tof = torch.nanmean(slowness_samples, dim=0) * path_length

    # F-number 孔径约束：限制有效的发射/接收角度
    # |2 * fnum * dx| <= dz 等价于 arctan(dx/dz) <= 1/(2*fnum)
    fnum_valid = torch.abs(2 * fnum * dx) <= dz

    # 最小孔径约束：即使 F-number 限制很严格，也保留至少 Dmin 宽的孔径
    Dmin_valid = (dz < Dmin * fnum) & (dx < Dmin / 2)

    # 合并两个约束
    valid = fnum_valid | Dmin_valid

    # 无效区域赋值为 -10（一个很大的负飞行时间），
    # 后续 DAS 插值时这些位置将产生零值
    tof_valid = torch.where(valid, tof, torch.ones_like(tof))
    tof = torch.where(valid, tof_valid, -10 * torch.ones_like(tof))

    return tof


# ==============================================================================
# 第四部分：损失函数（Loss Functions）
# ==============================================================================
# 这些函数用于评估超声图像的聚焦质量。在声速优化过程中，
# 优化器通过最小化这些损失函数来找到正确的声速分布。


def lag_one_coherence(iq, t_tx, t_rx, fs, fd):
    """滞后一相干性（Lag-One Coherence, LOC）。

    参考文献: DOI: 10.1109/TUFFC.2018.2855653

    LOC 衡量接收孔径上信号的空间相干性：相邻阵元接收信号的归一化互相关。
    LOC 接近 1 表示良好聚焦（信号主导），接近 0 表示散焦（噪声主导）。

    计算方法：
      1. 对每个接收阵元独立进行 DAS 波束合成（使用单位矩阵 A）
      2. 计算相邻阵元信号的归一化互相关系数 NCC

    Args:
        iq:    IQ 数据，形状 [nrx, ntx, n_samples]
        t_tx:  发射延时，形状 [ntx, *pixel_dims]
        t_rx:  接收延时，形状 [nrx, *pixel_dims]
        fs:    采样频率 (Hz)
        fd:    解调频率 (Hz)

    Returns:
        归一化互相关系数图，形状 [*pixel_dims]，值域 [0, 1]
    """
    # 交换维度使接收阵元在第一维（便于 DAS 处理）
    iq = iq.permute(1, 0, 2)
    # 使用单位矩阵进行 DAS：保留每个接收阵元的独立结果
    rxdata = das(iq, t_rx, t_tx, fs, fd,
                 torch.eye(iq.shape[0], dtype=iq.dtype, device=iq.device))
    # 相邻阵元的互相关
    xy = torch.real(torch.nansum(rxdata[:-1] * torch.conj(rxdata[1:]), dim=0))
    xx = torch.nansum(torch.abs(rxdata[:-1]) ** 2, dim=0)
    yy = torch.nansum(torch.abs(rxdata[1:]) ** 2, dim=0)
    ncc = xy / torch.sqrt(xx * yy)
    return ncc


def coherence_factor(iq, t_tx, t_rx, fs, fd):
    """相干因子（Coherence Factor, CF）。

    参考文献: DOI: 10.1121/1.410562

    CF 衡量接收孔径上信号的相干叠加与非相干叠加的比值。
    CF = |Σ s_i| / Σ |s_i|，其中 s_i 是各阵元的聚焦信号。
    CF = 1 表示完美聚焦（所有信号同相），CF << 1 表示存在像差。

    Args:
        iq:    IQ 数据，形状 [nrx, ntx, n_samples]
        t_tx:  发射延时
        t_rx:  接收延时
        fs:    采样频率 (Hz)
        fd:    解调频率 (Hz)

    Returns:
        相干因子图，形状 [*pixel_dims]，值域 [0, 1]
    """
    iq = iq.permute(1, 0, 2)
    rxdata = das(iq, t_rx, t_tx, fs, fd,
                 torch.eye(iq.shape[0], dtype=iq.dtype, device=iq.device))
    # 相干叠加的幅值 vs 非相干叠加
    numerator = torch.abs(torch.nansum(rxdata, dim=0))
    denominator = torch.nansum(torch.abs(rxdata), dim=0)
    return numerator / denominator


def speckle_brightness(iq, t_tx, t_rx, fs, fd):
    """散斑亮度（Speckle Brightness, SB）。

    参考文献: DOI: 10.1121/1.397889

    散斑亮度是 DAS 输出幅值的均值。正确聚焦的图像散斑亮度更高，
    因为信号相干叠加会增强亮度。

    Args:
        iq:    IQ 数据
        t_tx:  发射延时
        t_rx:  接收延时
        fs:    采样频率 (Hz)
        fd:    解调频率 (Hz)

    Returns:
        标量，散斑亮度值
    """
    return torch.nanmean(torch.abs(das(iq, t_tx, t_rx, fs, fd)))


def total_variation(c):
    """全变分正则化（Total Variation Regularization）。

    计算声速图在 x 和 z 方向上差分的平方均值之和。
    作为正则化项，鼓励声速图平滑，防止优化过程中出现过多噪声。

    TV(c) = mean((∂c/∂x)²) + mean((∂c/∂z)²)

    Args:
        c: 声速图，形状 [nx, nz]

    Returns:
        标量，全变分值
    """
    tvx = torch.nanmean(torch.square(torch.diff(c, dim=0)))  # x 方向
    tvz = torch.nanmean(torch.square(torch.diff(c, dim=1)))  # z 方向
    return tvx + tvz


def phase_error(iq, t_tx, t_rx, fs, fd, thresh=0.9):
    """相位误差（Phase Error）—— 基于平移孔径相关。

    参考文献:
      - Translated Transmit Apertures: DOI: 10.1109/58.585209
      - van Cittert-Zernike 定理: DOI: 10.1121/1.418235
      - 高质量相位估计: DOI: 10.1121/10.0000809
      - CUTE 算法: DOI: 10.1016/j.ultras.2020.106168

    核心思想：根据 van Cittert-Zernike 定理，具有共同中点的平移发射-接收
    子孔径对应产生完全相关的散斑信号。因此，这些信号之间的相位差可以用于
    估计像差引起的相位误差。

    算法步骤：
      1. 构建发射和接收子孔径掩码（带状对角矩阵）
      2. 对每个子孔径进行 DAS 波束合成
      3. 计算相邻共中点子孔径信号的互相关
      4. 使用相关系数阈值筛选高质量估计
      5. 提取相位差作为相位误差

    Args:
        iq:      IQ 数据，形状 [nrx, ntx, n_samples]
        t_tx:    发射延时
        t_rx:    接收延时
        fs:      采样频率 (Hz)
        fd:      解调频率 (Hz)
        thresh:  相关系数阈值（默认 0.9），低于此值的估计被丢弃

    Returns:
        相位误差矩阵 dphi，形状 [n_patches, n_subap-1, n_subap-1]
    """
    nrx, ntx, nsamps = iq.shape

    # 构建子孔径掩码：每个子孔径包含 2*halfsa+1 = 17 个相邻阵元
    halfsa = 8   # 子孔径半宽度
    dx = 1       # 子孔径滑动步长

    # 构建带状对角掩码矩阵
    mask = np.zeros((nrx, ntx))
    for diag in range(-halfsa, halfsa + 1):
        mask = mask + np.diag(np.ones((ntx - abs(diag),)), diag)
    # 裁剪行以获得有效的子孔径
    mask = mask[halfsa: mask.shape[0] - halfsa: dx]

    # 发射子孔径矩阵（翻转）和接收子孔径矩阵
    At = torch.tensor(mask[::-1].copy(), dtype=iq.dtype, device=iq.device)
    Ar = torch.tensor(mask.copy(), dtype=iq.dtype, device=iq.device)

    # 对所有子孔径进行 DAS 波束合成
    iqfoc = das(iq, t_tx, t_rx, fs, fd, At, Ar)

    # 计算相邻共中点子孔径信号的互相关
    xy = iqfoc[:-1, :-1] * torch.conj(iqfoc[+1:, +1:])  # 互相关
    xx = iqfoc[:-1, :-1] * torch.conj(iqfoc[:-1, :-1])  # 自相关 1
    yy = iqfoc[+1:, +1:] * torch.conj(iqfoc[+1:, +1:])  # 自相关 2

    # 排除只有一个信号（另一个为零）的相关
    valid_pair = (iqfoc[:-1, :-1] != 0) & (iqfoc[1:, 1:] != 0)
    xy = torch.where(valid_pair, xy, torch.zeros_like(xy))
    xx = torch.where(valid_pair, xx, torch.zeros_like(xx))
    yy = torch.where(valid_pair, yy, torch.zeros_like(yy))

    # 在核（kernel）维度上求和以提高信噪比
    xy = torch.sum(xy, dim=-1)
    xx = torch.sum(xx, dim=-1)
    yy = torch.sum(yy, dim=-1)

    # 相关系数平方，用于质量筛选
    cc_squared = torch.square(torch.abs(xy)) / (torch.abs(xx) * torch.abs(yy))
    high_quality = cc_squared > thresh * thresh
    xy = torch.where(high_quality, xy, torch.zeros_like(xy))

    # 将反对角排列转换为对角排列，并构建对称矩阵
    xy = xy.flip(0)  # 反对角 → 对角
    xy = xy.reshape(*xy.shape[:2], -1)
    xy = xy.permute(2, 0, 1)  # 将 patch 维度移到最外层

    # 构建 Hermite 对称矩阵：上三角 + 下三角的共轭转置
    xy = torch.triu(xy) + torch.conj(torch.tril(xy)).permute(0, 2, 1)

    # 提取相位（角度）作为相位误差
    dphi = torch.angle(xy)
    return dphi


# ==============================================================================
# 第五部分：可视化工具（Visualization Utilities）
# ==============================================================================


def imagesc(xc, y, img, dynamic_range, **kwargs):
    """MATLAB 风格的 imagesc 函数。

    以正确的坐标范围和颜色映射显示 2D 图像，自动添加颜色条。

    Args:
        xc:            x 轴坐标向量
        y:             y 轴坐标向量
        img:           2D 图像数组
        dynamic_range: 颜色映射范围 [vmin, vmax]
        **kwargs:      传递给 plt.imshow 的额外参数

    Returns:
        matplotlib AxesImage 对象
    """
    dx = xc[1] - xc[0]
    dy = y[1] - y[0]
    # 计算图像像素边缘的空间范围（而非中心位置）
    extent = [xc[0] - dx / 2, xc[-1] + dx / 2, y[-1] + dy / 2, y[0] - dy / 2]
    im = plt.imshow(img, vmin=dynamic_range[0], vmax=dynamic_range[1],
                    extent=extent, **kwargs)
    plt.colorbar()
    return im


def plot_loss_curves(c0, dsb, dlc, dcf, dpe, sample):
    """绘制各损失函数随全局声速变化的曲线。

    用于初始化阶段，可视化不同聚焦质量指标与声速的关系，
    帮助选择最优的初始声速。

    Args:
        c0:     声速扫描值数组，形状 [n_speeds]
        dsb:    散斑亮度损失
        dlc:    滞后一相干性损失
        dcf:    相干因子损失
        dpe:    相位误差损失
        sample: 样本名称（用于标题）
    """
    plt.clf()
    plt.plot(c0, dsb, label="Speckle Brightness")
    plt.plot(c0, dlc, label="Lag One Coherence")
    plt.plot(c0, dcf, label="Coherence Factor")
    plt.plot(c0, dpe / 10, label="Phase Error (÷10)")  # 缩放 10 倍以便可视化比较
    plt.grid()
    plt.xlabel("Global sound speed (m/s)")
    plt.ylabel("Loss function")
    plt.title(sample)
    plt.legend()
    plt.savefig(f"images/losses_{sample}.png")
    plt.savefig("scratch.png")
    plt.clf()
