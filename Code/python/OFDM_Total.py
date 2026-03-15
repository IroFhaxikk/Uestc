"""
OFDM_full_link.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
完整 OFDM 通信链路仿真

【系统框图】
  发射端：比特流 → 16-QAM → 导频插入 → IFFT → CP → 上采样 → RRC → 射频上变频
  信道：  2径随机瑞利块衰落（每块独立）+ AWGN
  接收端：射频下变频 → RRC → 降采样 → 去CP → FFT
          → 信道估计（LS线性 / LS样条 / MMSE）
          → 信道均衡（ZF / MMSE均衡 / 三抽头时域均衡）
          → 16-QAM解调 → BER统计

【均衡算法说明】
  ① ZF（迫零）        : Y_eq[k] = Y[k] / H_est[k]
                         完全消除信道，深衰落处噪声放大严重
  ② MMSE均衡          : Y_eq[k] = Y[k]·H*[k] / (|H[k]|² + 1/SNR)
                         折中消除失真与抑制噪声，深衰落处鲁棒
  ③ 三抽头时域均衡    : 时域 FIR 滤波 w = [w₋₁, w₀, w₁]，
                         权向量由 MMSE 准则（Wiener-Hopf）求解
                         适用于无 CP 或 CP 不足场景，与频域均衡对比

【输出图表】
  图1 BER vs Eb/N0（9条曲线：完美均衡 + 3估计 × 2频域均衡 + 三抽头）
  图2 MSE vs Eb/N0（信道估计质量）
  图3 信道频率响应可视化（3种估计对比）
  图4 均衡前后星座图（5种状态对比）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

matplotlib.use('TkAgg')   # macOS 可改 'MacOSX'，Linux 改 'Qt5Agg'


# ════════════════════════════════════════════════════════════════════════════
# § 0  中文字体（跨平台）
# ════════════════════════════════════════════════════════════════════════════
def _find_cjk_font():
    for p in [
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/System/Library/Fonts/Supplemental/Songti.ttc',
        '/Library/Fonts/Arial Unicode MS.ttf',
        os.path.expanduser('~/Library/Fonts/NotoSansCJK-Regular.ttc'),
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
    ]:
        if os.path.exists(p):
            return p
    return None

_cjk = _find_cjk_font()
if _cjk:
    fm.fontManager.addfont(_cjk)
    matplotlib.rcParams['font.family'] = [
        fm.FontProperties(fname=_cjk).get_name(), 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ════════════════════════════════════════════════════════════════════════════
# § 1  系统参数（统一集中定义，修改方便）
# ════════════════════════════════════════════════════════════════════════════

# ─ 调制 ──────────────────────────────────────────────────────────────────
M        = 16                       # QAM 阶数
k_bits   = int(np.log2(M))         # 每符号比特数 = 4
N        = 64                       # 子载波数
CP       = 16                       # 循环前缀长度（≥ 信道最大时延）

# ─ 射频链路 ──────────────────────────────────────────────────────────────
UP       = 120                      # 上采样倍数
BETA     = 0.4                      # RRC 滚降系数
FC       = 1.6e9                    # 载波频率 1.6 GHz
FS       = UP * 1e9                 # 采样率 120 GHz

# ─ 导频（梳状，与 matlab_ref 风格一致）──────────────────────────────────
NPS      = 8                        # 导频间隔（每 8 个子载波 1 个导频）
P_LOC    = list(range(0, N, NPS))   # 导频下标列表，共 8 个
D_LOC    = [k for k in range(N) if k not in P_LOC]  # 数据下标，共 56 个
NP       = len(P_LOC)
ND       = len(D_LOC)

# ─ 2径瑞利信道（matlab_ref 风格，每块独立生成）──────────────────────────
#   h[0] ~ CN(0,1)，h[1] ~ CN(0, 0.25)（第二径功率 -6 dB）
CH_LEN   = 2                        # 信道径数（用于 DFT 截断）

# ─ 仿真配置 ──────────────────────────────────────────────────────────────
EbN0_dB  = np.arange(0, 30, 2)     # Eb/N0 扫描范围
NBLK     = 300                      # 每个 SNR 点的仿真块数
VIS_IDX  = len(EbN0_dB) // 2       # 取中间 SNR 点做可视化

# ─ SNR 换算 ──────────────────────────────────────────────────────────────
# SNR（子载波级）= Eb/N0 + 10log10(k_bits) + 导频开销修正
_pilot_loss = 10 * np.log10(ND / N)       # 导频占用开销（负值，约 -0.58 dB）
SNR_dB   = EbN0_dB + 10*np.log10(k_bits) + _pilot_loss

# ─ 16-QAM 映射表 ─────────────────────────────────────────────────────────
_MAP = {
    (0,0,0,0):-3-3j,(0,0,0,1):-3-1j,(0,0,1,0):-3+3j,(0,0,1,1):-3+1j,
    (0,1,0,0):-1-3j,(0,1,0,1):-1-1j,(0,1,1,0):-1+3j,(0,1,1,1):-1+1j,
    (1,0,0,0): 3-3j,(1,0,0,1): 3-1j,(1,0,1,0): 3+3j,(1,0,1,1): 3+1j,
    (1,1,0,0): 1-3j,(1,1,0,1): 1-1j,(1,1,1,0): 1+3j,(1,1,1,1): 1+1j,
}
_SCALE  = np.sqrt(np.mean([np.abs(v)**2 for v in _MAP.values()]))  # ≈ √10
CONST   = np.array(list(_MAP.values())) / _SCALE   # 归一化星座点
BITMAP  = list(_MAP.keys())


# ════════════════════════════════════════════════════════════════════════════
# § 2  射频链路工具
# ════════════════════════════════════════════════════════════════════════════

def make_rrc(beta, span, sps):
    """根升余弦（RRC）滤波器，单位能量归一化。"""
    delay = span * sps / 2
    t = np.arange(-delay, delay + 1) / sps
    t[np.abs(t) < 1e-12] = 0.0
    num   = np.sin(np.pi*t*(1-beta)) + 4*beta*t*np.cos(np.pi*t*(1+beta))
    denom = np.pi * t * (1 - (4*beta*t)**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = num / denom
    h[t == 0] = 1.0 - beta + 4*beta/np.pi
    if beta != 0:
        idx = np.abs(np.abs(t) - 1/(4*beta)) < 1e-5
        h[idx] = (beta/np.sqrt(2)) * (
            (1+2/np.pi)*np.sin(np.pi/(4*beta)) +
            (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    return h / np.sqrt(np.sum(h**2))

RC = make_rrc(BETA, 4, UP)   # 全局 RRC 系数


def rf_up(x_bb, fs, fc):
    """
    射频上变频：复基带 → 实数带通信号。
    s_rf(t) = Re{ x_bb(t) · √2·e^{j2πfc·t} }
    """
    t   = np.arange(len(x_bb)) / fs
    car = np.sqrt(2) * np.exp(1j * 2*np.pi * fc * t)
    return np.real(x_bb * car), car


def rf_down(r_rf, car):
    """
    射频下变频：实数带通 → 复基带。
    x_bb(t) = r_rf(t) · Re{car} - j·r_rf(t)·Im{car}
    """
    return r_rf * np.real(car) + 1j * r_rf * (-np.imag(car))


# ════════════════════════════════════════════════════════════════════════════
# § 3  信道生成
# ════════════════════════════════════════════════════════════════════════════

def make_channel_2tap():
    """
    生成 2径随机瑞利信道（matlab_ref 风格）。
      h[0] ~ CN(0, 1)      第一径，主径
      h[1] ~ CN(0, 0.25)   第二径，-6 dB
    每次调用独立生成，实现块衰落。
    """
    h = np.array([
        np.random.randn() + 1j*np.random.randn(),           # CN(0,1)
        (np.random.randn() + 1j*np.random.randn()) / 2.0    # CN(0,0.25)
    ])
    return h


# ════════════════════════════════════════════════════════════════════════════
# § 4  信道估计
# ════════════════════════════════════════════════════════════════════════════

def ce_ls(Y, Xp, p_loc, N, kind):
    """
    LS 信道估计 + 插值。
    导频处：H_ls[p] = Y[p] / Xp
    其余处：interp1d 插值（linear 或 cubic 样条）

    复杂度低，不需要信道统计先验，工程实用。
    缺点：噪声大时精度差，样条在端点处可能过拟合。
    """
    p_loc  = np.array(p_loc)
    H_p    = Y[p_loc] / Xp                            # 导频处直接相除
    interp = interp1d(p_loc, H_p, kind=kind, fill_value='extrapolate')
    return interp(np.arange(N))


def ce_mmse(Y, Xp, p_loc, N, Nps, h_true, snr_lin):
    """
    MMSE 信道估计（Wiener 滤波）。
    利用指数功率延迟谱先验构建最优线性估计器：
      H_mmse = Rhp · (Rpp + I/SNR)^{-1} · H_tilde
    tau_rms 由真实 h 计算（上帝视角，与 matlab_ref 一致）。

    优点：噪声大时有正则化效果，全局平滑，MSE 最小。
    """
    p_loc   = np.array(p_loc)
    Np      = len(p_loc)
    H_tilde = Y[p_loc] / Xp                           # 导频处 LS 初估

    # 计算 RMS 时延扩展
    k_h     = np.arange(len(h_true), dtype=float)
    hh      = np.dot(h_true, h_true.conj()).real
    tmp     = h_true * h_true.conj() * k_h
    r1      = np.sum(tmp).real / hh                    # 平均时延
    r2      = np.dot(tmp, k_h).real / hh               # 均方时延
    tau_rms = np.sqrt(max(r2 - r1**2, 0))

    # 指数谱相关模型
    j2pi = 1j * 2 * np.pi * tau_rms / N

    # 互相关矩阵 Rhp (N × Np)
    K1  = np.tile(np.arange(N)[:, None],  (1, Np))
    K2  = np.tile(np.arange(Np)[None, :], (N, 1))
    Rhp = 1.0 / (1 + j2pi * (K1 - K2 * Nps))

    # 自相关矩阵 Rpp (Np × Np) + 噪声正则项
    K3  = np.tile(np.arange(Np)[:, None], (1, Np))
    K4  = np.tile(np.arange(Np)[None, :], (Np, 1))
    Rpp = 1.0 / (1 + j2pi * Nps * (K3 - K4)) + np.eye(Np) / snr_lin

    return Rhp @ np.linalg.solve(Rpp, H_tilde)


def dft_denoise(H_est, ch_len):
    """
    DFT 降噪：IFFT → 截断前 ch_len 个时域抽头 → FFT。
    噪声被压缩 N/ch_len 倍（≈ 10·log10(N/L) dB）。
    """
    h_t = np.fft.ifft(H_est)
    return np.fft.fft(h_t[:ch_len], len(H_est))


# ════════════════════════════════════════════════════════════════════════════
# § 5  信道均衡
# ════════════════════════════════════════════════════════════════════════════

def eq_zf(Y, H):
    """
    ZF（迫零）频域均衡。
    ─────────────────────────────────────────────
    原理：Y[k] = H[k]·X[k] + N[k]
          直接除以 H[k] → X̂[k] = X[k] + N[k]/H[k]
    问题：|H[k]| → 0 时噪声项 N[k]/H[k] → ∞
          深衰落子载波 SNR 急剧恶化
    适用：高 SNR、信道无深衰落场景
    """
    return Y / H


def eq_mmse(Y, H, snr_lin):
    """
    MMSE 频域均衡。
    ─────────────────────────────────────────────
    原理：最小化 E[|X̂[k] - X[k]|²]
          最优权向量：Q[k] = H*[k] / (|H[k]|² + 1/SNR)
    分析：分母加了正则项 1/SNR，避免深衰落处除零
          高 SNR → 1/SNR→0 → 退化为 ZF
          低 SNR → 1/SNR 主导 → 幅度缩放，保噪声
    适用：全 SNR 范围，深衰落信道的标准选择
    """
    return Y * H.conj() / (np.abs(H)**2 + 1.0 / snr_lin)


def eq_3tap_mmse(Y, H_est, snr_lin, N):
    """
    三抽头时域 MMSE 均衡（FIR-MMSE）。
    ─────────────────────────────────────────────
    思路：
      1. 从 H_est 还原时域信道 h_est = IFFT(H_est)，取前 3 个抽头
      2. 构造 3×3 循环卷积矩阵 A（A[i,j] = h[(i-j) mod 3]）
      3. 用 Wiener-Hopf MMSE 准则求权向量：
            W = (A^H·A + σ²I)^{-1}·A^H
         取中心行（对应当前时刻输出），得 w = [w₋₁, w₀, w₁]
      4. 对时域接收信号循环滤波：
            y_eq[n] = w₋₁·y[n-1] + w₀·y[n] + w₁·y[n+1]
      5. 再 FFT 转回频域提取数据

    物理含义：
      FIR 滤波器近似信道的逆，三抽头是最简单的非平凡 FIR 均衡器。
      加正则项 σ²I 避免深衰落处噪声放大（比 FIR-ZF 更鲁棒）。

    在 OFDM+CP 场景：CP 已消除 ISI，频域单抽头是最优解，
      三抽头主要用于学习对比，或处理 CP 不足时的残余 ISI。
    """
    # 从估计频响还原时域，取前 3 抽头
    h3 = np.fft.ifft(H_est)[:3].copy()

    # 构造 3×3 循环卷积矩阵
    A = np.array([[h3[(i - j) % 3] for j in range(3)] for i in range(3)])

    # Wiener-Hopf MMSE 解，取中心行（索引1，对应当前时刻）
    AhA = A.conj().T @ A
    W   = np.linalg.solve(AhA + np.eye(3) / snr_lin, A.conj().T)
    w   = W[1, :]                                      # shape (3,)

    # 时域循环滤波
    y    = np.fft.ifft(Y)
    y_eq = np.array([
        w[0]*y[(n-1) % N] + w[1]*y[n] + w[2]*y[(n+1) % N]
        for n in range(N)
    ])

    return np.fft.fft(y_eq)                            # 转回频域


# ════════════════════════════════════════════════════════════════════════════
# § 6  解调
# ════════════════════════════════════════════════════════════════════════════

def demodulate(Y_eq, d_loc):
    """最近邻硬判决解调，返回比特列表。"""
    rx  = Y_eq[d_loc]
    out = []
    for sym in rx:
        out.extend(BITMAP[np.argmin(np.abs(sym - CONST))])
    return out


# ════════════════════════════════════════════════════════════════════════════
# § 7  主仿真循环
# ════════════════════════════════════════════════════════════════════════════

# BER 结果：9 条曲线
#  0  完美ZF（上界）
#  1  LS线性  + ZF
#  2  LS线性  + MMSE均衡
#  3  LS样条  + ZF
#  4  LS样条  + MMSE均衡
#  5  MMSE估计 + ZF
#  6  MMSE估计 + MMSE均衡
#  7  MMSE估计 + 三抽头MMSE
#  8  （预留）完美三抽头MMSE（用真实H，作为三抽头上界）
NCURVES = 9
BER  = np.zeros((len(EbN0_dB), NCURVES))
MSE  = np.zeros((len(EbN0_dB), 3))    # LS线性 / LS样条 / MMSE估计 的 MSE

# 可视化数据容器
vis_done      = False
vis_H_true_dB = None
vis_H_ests_dB = {}          # 三种估计的频响（dB）
vis_syms      = {}          # 各均衡方案的数据子载波复数点

print("=" * 72)
print("  OFDM 完整通信链路  |  16-QAM  |  N=64  |  CP=16  |  2径瑞利信道")
print(f"  导频 Nps={NPS}，{NP}/{N} 个导频，{ND} 个数据子载波")
print(f"  RRC β={BETA}  fc={FC/1e9:.1f} GHz  fs={FS/1e9:.0f} GHz")
print(f"  均衡算法：ZF / MMSE / 三抽头MMSE")
print("=" * 72)

hdr = (f"\n{'Eb/N0':>5} | {'完美ZF':>8} | {'LS+ZF':>7} | {'LS+MMSE':>8} | "
       f"{'SPL+ZF':>7} | {'SPL+MMSE':>9} | {'CE+ZF':>7} | "
       f"{'CE+MMSE':>8} | {'CE+3MMSE':>9} | {'真实+3MMSE':>11}")
print(hdr)
print("-" * len(hdr))

np.random.seed(42)

for ji, snr_db in enumerate(SNR_dB):
    snr_lin = 10**(snr_db / 10)
    errs    = np.zeros(NCURVES, dtype=int)
    nbits   = 0
    mse_acc = np.zeros(3)

    for blk in range(NBLK):

        # ── § 7.1  发射端 ────────────────────────────────────────────────

        # 导频符号（全 +1，收发双方均已知）
        Xp = np.ones(NP, dtype=float)

        # 随机比特 → 16-QAM 调制
        tx_bits  = np.random.randint(0, 2, ND * k_bits)
        tx_syms  = np.array([
            _MAP[tuple(tx_bits[i:i+k_bits])] / _SCALE
            for i in range(0, ND*k_bits, k_bits)
        ])

        # 子载波复用：导频 + 数据
        X                = np.zeros(N, dtype=complex)
        X[P_LOC]         = Xp
        X[D_LOC]         = tx_syms

        # IFFT（基带 OFDM 调制）+ 循环前缀
        x_time = np.fft.ifft(X) * np.sqrt(N)
        x_cp   = np.concatenate([x_time[-CP:], x_time])   # (N+CP,)

        # 上采样 → RRC 脉冲成型（带限 + 无 ISI）
        x_up       = np.zeros(len(x_cp) * UP, dtype=complex)
        x_up[::UP] = x_cp
        x_rrc      = np.convolve(x_up, RC, mode='full')

        # 射频上变频
        tx_rf, _ = rf_up(x_rrc, FS, FC)

        # ── § 7.2  2径瑞利信道 ──────────────────────────────────────────

        h      = make_channel_2tap()           # 每块独立，实现块衰落
        H_true = np.fft.fft(h, N)             # 真实频率响应

        # 基带等效多径卷积 → 重新上采样成型 → 射频
        x_ch       = fftconvolve(x_cp, h)
        x_ch_up    = np.zeros(len(x_ch) * UP, dtype=complex)
        x_ch_up[::UP] = x_ch
        x_ch_rrc   = fftconvolve(x_ch_up, RC)

        tx_rf2, car2 = rf_up(x_ch_rrc, FS, FC)

        # AWGN（从射频信号测量实际功率，保证 SNR 精确）
        pwr       = np.mean(tx_rf2**2) * UP * 0.5
        noise_std = np.sqrt(pwr / snr_lin)
        rx_rf     = tx_rf2 + np.random.normal(0, noise_std, len(tx_rf2))

        # ── § 7.3  接收端 ─────────────────────────────────────────────

        # 射频下变频 → RRC 匹配滤波
        rx_bb  = rf_down(rx_rf, car2)
        rx_rrc = fftconvolve(rx_bb, RC)

        # 降采样（去掉 RRC 两端拖尾）
        Lch    = len(x_ch)
        start  = len(RC) - 1
        rx_dn  = rx_rrc[start : start + Lch*UP : UP]

        # 去循环前缀 → FFT（频域信道模型：Y[k] = H[k]·X[k] + N[k]）
        Y = np.fft.fft(rx_dn[CP : CP + N]) / np.sqrt(N)

        # ── § 7.4  信道估计 ───────────────────────────────────────────

        # LS 线性插值（+ DFT降噪）
        H_ls_lin = dft_denoise(
            ce_ls(Y, Xp, P_LOC, N, 'linear'), CH_LEN)

        # LS 样条插值（+ DFT降噪）
        H_ls_spl = dft_denoise(
            ce_ls(Y, Xp, P_LOC, N, 'cubic'), CH_LEN)

        # MMSE 估计（+ DFT降噪）
        H_mmse_e = dft_denoise(
            ce_mmse(Y, Xp, P_LOC, N, NPS, h, snr_lin), CH_LEN)

        # 估计 MSE 统计
        mse_acc[0] += np.mean(np.abs(H_true - H_ls_lin)**2)
        mse_acc[1] += np.mean(np.abs(H_true - H_ls_spl)**2)
        mse_acc[2] += np.mean(np.abs(H_true - H_mmse_e)**2)

        # ── § 7.5  信道均衡（9种方案）──────────────────────────────────

        Y_eqs = [
            eq_zf(Y, H_true),                              # 0 完美ZF（上界）
            eq_zf(Y, H_ls_lin),                            # 1 LS线性 + ZF
            eq_mmse(Y, H_ls_lin, snr_lin),                 # 2 LS线性 + MMSE
            eq_zf(Y, H_ls_spl),                            # 3 LS样条 + ZF
            eq_mmse(Y, H_ls_spl, snr_lin),                 # 4 LS样条 + MMSE
            eq_zf(Y, H_mmse_e),                            # 5 MMSE估计 + ZF
            eq_mmse(Y, H_mmse_e, snr_lin),                 # 6 MMSE估计 + MMSE
            eq_3tap_mmse(Y, H_mmse_e, snr_lin, N),         # 7 MMSE估计 + 三抽头
            eq_3tap_mmse(Y, H_true, snr_lin, N),           # 8 真实H + 三抽头（上界）
        ]

        # ── § 7.6  解调 + BER ────────────────────────────────────────

        for mi, Y_eq in enumerate(Y_eqs):
            db = demodulate(Y_eq, D_LOC)
            errs[mi] += np.sum(np.array(db) != tx_bits)
        nbits += ND * k_bits

        # ── § 7.7  保存可视化数据（中间 SNR 首块）────────────────────

        if not vis_done and ji == VIS_IDX and blk == 0:
            vis_H_true_dB = 20*np.log10(np.abs(H_true) + 1e-12)
            vis_H_ests_dB = {
                'LS线性（+DFT）': 20*np.log10(np.abs(H_ls_lin) + 1e-12),
                'LS样条（+DFT）': 20*np.log10(np.abs(H_ls_spl) + 1e-12),
                'MMSE（+DFT）':   20*np.log10(np.abs(H_mmse_e) + 1e-12),
            }
            vis_syms = {
                '均衡前':           Y[D_LOC],
                '完美ZF':           Y_eqs[0][D_LOC],
                'MMSE估计+ZF':      Y_eqs[5][D_LOC],
                'MMSE估计+MMSE均衡':Y_eqs[6][D_LOC],
                'MMSE估计+三抽头':  Y_eqs[7][D_LOC],
            }
            vis_done = True

    BER[ji] = errs / nbits
    MSE[ji] = mse_acc / NBLK

    v = BER[ji]
    print(f"{EbN0_dB[ji]:>5} | "
          f"{v[0]:>8.5f} | {v[1]:>7.5f} | {v[2]:>8.5f} | "
          f"{v[3]:>7.5f} | {v[4]:>9.5f} | {v[5]:>7.5f} | "
          f"{v[6]:>8.5f} | {v[7]:>9.5f} | {v[8]:>11.5f}")


# ════════════════════════════════════════════════════════════════════════════
# § 8  绘图
# ════════════════════════════════════════════════════════════════════════════

IDEAL_PTS = np.array(list(_MAP.values())) / _SCALE
K_AX      = np.arange(N)
VIS_EbN0  = EbN0_dB[VIS_IDX]

# ── 图1：BER vs Eb/N0 ────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 7))

_ber_cfg = [
    # (标签,                颜色,      标记, 线型,  线宽)
    ('完美ZF（性能上界）',  '#0D47A1', 'D',  '-',   2.5),
    ('LS线性 + ZF',         '#2E7D32', 's',  '--',  1.6),
    ('LS线性 + MMSE均衡',   '#66BB6A', 's',  ':',   1.6),
    ('LS样条 + ZF',         '#E65100', '^',  '--',  1.6),
    ('LS样条 + MMSE均衡',   '#FFA726', '^',  ':',   1.6),
    ('MMSE估计 + ZF',       '#880E4F', 'o',  '--',  1.8),
    ('MMSE估计 + MMSE均衡', '#E91E63', 'o',  '-',   2.2),
    ('MMSE估计 + 三抽头MMSE','#4A148C','v',  '--',  1.6),
    ('真实H + 三抽头MMSE',  '#7B1FA2', 'v',  ':',   1.6),
]
for mi, (lbl, col, mk, ls, lw) in enumerate(_ber_cfg):
    ax1.semilogy(EbN0_dB, np.maximum(BER[:, mi], 1e-6),
                 color=col, marker=mk, ls=ls, lw=lw, ms=7, label=lbl)

ax1.set_xlabel('Eb/N0 (dB)', fontsize=12)
ax1.set_ylabel('误码率 BER', fontsize=12)
ax1.set_title(
    '16-QAM OFDM 误码率 —— 信道估计 × 均衡算法全对比\n'
    '2径瑞利块衰落信道  |  RRC成型  |  射频链路  |  DFT降噪',
    fontsize=11
)
ax1.legend(fontsize=9, loc='lower left', ncol=2)
ax1.grid(True, which='both', alpha=0.3)
ax1.set_ylim(1e-4, 1.0)
fig1.tight_layout()


# ── 图2：MSE vs Eb/N0（信道估计质量）────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 6))

_mse_cfg = [
    ('LS线性（+DFT）', '#2E7D32', 's'),
    ('LS样条（+DFT）', '#E65100', '^'),
    ('MMSE（+DFT）',   '#E91E63', 'o'),
]
for mi, (lbl, col, mk) in enumerate(_mse_cfg):
    ax2.semilogy(EbN0_dB, np.maximum(MSE[:, mi], 1e-10),
                 color=col, marker=mk, lw=1.8, ms=7, label=lbl)

ax2.set_xlabel('Eb/N0 (dB)', fontsize=12)
ax2.set_ylabel('信道估计 MSE', fontsize=12)
ax2.set_title('信道估计 MSE vs Eb/N0\n（三种估计方法，均经过 DFT 降噪）', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, which='both', alpha=0.3)
fig2.tight_layout()


# ── 图3：信道频率响应可视化（3估计 × 1行）────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle(
    f'信道频率响应估计对比（Eb/N0={VIS_EbN0} dB，2径瑞利信道，已DFT降噪）',
    fontsize=11, fontweight='bold'
)
for col_i, (lbl, H_dB) in enumerate(vis_H_ests_dB.items()):
    ax = axes3[col_i]
    ax.plot(K_AX, vis_H_true_dB, 'b', lw=2, label='真实 H')
    ax.plot(K_AX, H_dB, 'r--', lw=1.5, label=lbl)
    ax.scatter(P_LOC, [H_dB[p] for p in P_LOC],
               color='k', s=30, zorder=5, label='导频估计点')
    ax.set_title(lbl, fontsize=10, fontweight='bold')
    ax.set_xlabel('子载波序号')
    ax.set_ylabel('|H[k]| (dB)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig3.tight_layout()


# ── 图4：均衡前后星座图（1行 5列）────────────────────────────────────────
_sym_cfg = [
    ('均衡前\n（信道畸变）',      '#546E7A'),
    ('完美ZF\n（性能上界）',      '#0D47A1'),
    ('MMSE估计\n+ ZF均衡',        '#880E4F'),
    ('MMSE估计\n+ MMSE均衡',      '#E91E63'),
    ('MMSE估计\n+ 三抽头MMSE',    '#4A148C'),
]
sym_keys = list(vis_syms.keys())

fig4, axes4 = plt.subplots(1, 5, figsize=(18, 4))
fig4.suptitle(
    f'均衡前后星座图（Eb/N0={VIS_EbN0} dB，2径瑞利信道）',
    fontsize=12, fontweight='bold'
)
for col_i, ((title, col_c), key) in enumerate(zip(_sym_cfg, sym_keys)):
    ax = axes4[col_i]
    s  = np.array(vis_syms[key])
    ax.scatter(s.real, s.imag, s=5, color=col_c, alpha=0.5)
    ax.scatter(IDEAL_PTS.real, IDEAL_PTS.imag,
               s=60, color='k', marker='+', lw=1.5, zorder=5)
    ax.set_title(title, fontsize=9, fontweight='bold')
    lim = 3.5 if col_i == 0 else 2.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')

fig4.tight_layout()
plt.show()