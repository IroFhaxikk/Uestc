"""
OFDM_combined.py

融合版本：
  骨架来自 OFDM        —— 列矩阵处理 / RRC成型 / 射频上下变频 / Eb/N0 x轴
  信道来自 sim4        —— 5径固定功率时延瑞利信道（PowerdB/Delay 参数风格）
  估计来自 matlab_ref  —— LS线性 / LS样条 / MMSE（上帝视角 tau_rms）+ DFT降噪

输出四张图：
  图1 —— BER vs Eb/N0（完美均衡 / LS线性 / LS样条 / MMSE）
  图2 —— MSE vs Eb/N0（6条：3方法 × 有/无DFT降噪）
  图3 —— 首块信道频率响应对比（3×2子图）
  图4 —— 均衡前后星座图
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

matplotlib.use('TkAgg')   # macOS 可改 'MacOSX'，Linux 改 'Qt5Agg'

# ── 中文字体（跨平台自动查找）────────────────────────────────────────────────
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
# 工具函数
# ════════════════════════════════════════════════════════════════════════════

def rcosdesign(beta, span, sps):
    """根升余弦滤波器，单位能量归一化（对应 MATLAB rcosdesign）"""
    delay = span * sps / 2
    t = np.arange(-delay, delay + 1) / sps
    t[np.abs(t) < 1e-12] = 0.0
    num   = np.sin(np.pi*t*(1-beta)) + 4*beta*t*np.cos(np.pi*t*(1+beta))
    denom = np.pi*t*(1-(4*beta*t)**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = num / denom
    h[t == 0] = 1.0 - beta + 4*beta/np.pi
    if beta != 0:
        idx = np.abs(np.abs(t) - 1/(4*beta)) < 1e-5
        h[idx] = (beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))
                                    +(1-2/np.pi)*np.cos(np.pi/(4*beta)))
    return h / np.sqrt(np.sum(h**2))


def make_channel(PowerdB, Delay):
    """
    生成一次多径瑞利信道冲激响应（每块独立调用）
    PowerdB : 各径功率 [dB]，如 [0, -8, -17, -21, -25]
    Delay   : 各径时延 [采样]，如 [0, 3, 5, 6, 8]
    返回 h  : 长度为 Delay[-1]+1 的复数数组，非时延位置为 0
    """
    Power = 10**(np.array(PowerdB) / 10)       # dB → 线性功率
    Lch   = Delay[-1] + 1                       # 信道长度
    h     = np.zeros(Lch, dtype=complex)
    for d, p in zip(Delay, Power):
        # 每径增益服从 CN(0, p)，实部虚部各 N(0, p/2)
        h[d] = (np.random.randn() + 1j*np.random.randn()) * np.sqrt(p / 2)
    return h


# ════════════════════════════════════════════════════════════════════════════
# 信道估计函数
# ════════════════════════════════════════════════════════════════════════════

def ls_estimate(Y, Xp, pilot_loc, Nfft, kind='linear'):
    """
    LS 信道估计 + 插值
    Y         : FFT 后接收信号 (Nfft,)
    Xp        : 已知导频符号   (Np,)
    pilot_loc : 导频子载波下标列表
    kind      : 'linear' 线性 / 'cubic' 三次样条
    """
    pilot_loc = np.array(pilot_loc)
    H_ls = Y[pilot_loc] / Xp                          # 导频处直接相除 → LS 估计
    f    = interp1d(pilot_loc, H_ls, kind=kind,
                    fill_value='extrapolate')
    return f(np.arange(Nfft))                          # 插值到全部子载波


def mmse_estimate(Y, Xp, pilot_loc, Nfft, Nps, h_true, snr_lin):
    """
    MMSE 信道估计（上帝视角：用真实 h 计算 tau_rms）
    与 matlab_ref / sim4 的 MMSE_CE 函数逻辑一致

    h_true  : 真实信道冲激响应（含零填充的完整向量）
    snr_lin : 线性 SNR（注意不是 Eb/N0）
    """
    pilot_loc = np.array(pilot_loc)
    Np        = len(pilot_loc)
    H_tilde   = Y[pilot_loc] / Xp                     # 导频处 LS 初估

    # ── 用真实 h 计算 RMS 时延扩展（上帝视角，与 matlab_ref 一致）──────────
    k_h     = np.arange(len(h_true), dtype=float)
    hh      = np.dot(h_true, h_true.conj()).real       # 总功率
    tmp     = h_true * h_true.conj() * k_h
    r       = np.sum(tmp).real / hh                    # 平均时延
    r2      = np.dot(tmp, k_h).real / hh               # 均方时延
    tau_rms = np.sqrt(max(r2 - r**2, 0))               # RMS 时延扩展

    # ── 构建相关矩阵（指数功率延迟谱模型）─────────────────────────────────
    j2pi_tau_df = 1j * 2 * np.pi * tau_rms / Nfft

    # Rhp (Nfft × Np)：全部子载波 ↔ 导频位置的互相关
    K1  = np.tile(np.arange(Nfft)[:, None], (1, Np))
    K2  = np.tile(np.arange(Np)[None, :],   (Nfft, 1))
    Rhp = 1.0 / (1 + j2pi_tau_df * (K1 - K2 * Nps))

    # Rpp (Np × Np)：导频位置自相关 + 噪声正则项 I/snr
    K3  = np.tile(np.arange(Np)[:, None], (1, Np))
    K4  = np.tile(np.arange(Np)[None, :], (Np, 1))
    Rpp = 1.0 / (1 + j2pi_tau_df * Nps * (K3 - K4)) + np.eye(Np) / snr_lin

    return Rhp @ np.linalg.solve(Rpp, H_tilde)        # Wiener 滤波输出


def dft_denoise(H_est, channel_length):
    """
    DFT 降噪：IFFT → 截断前 channel_length 个时域抽头 → FFT
    原理：信道稀疏（L << N），噪声均匀铺散在 N 个时域位置，
    截断后噪声被压缩 N/L 倍（约 10*log10(N/L) dB）
    channel_length 应等于真实信道的有效径数
    """
    h_est = np.fft.ifft(H_est)
    return np.fft.fft(h_est[:channel_length], len(H_est))


# ════════════════════════════════════════════════════════════════════════════
# 系统参数
# ════════════════════════════════════════════════════════════════════════════

# ── 调制参数（来自 OFDM）─────────────────────────────────────────────────────
M               = 16               # 16-QAM
k_bits          = int(np.log2(M))  # 每符号比特数 = 4
numSubcarriers  = 64               # 子载波总数
cp_len          = 16               # 循环前缀长度

# ── 射频链路参数（来自 OFDM）─────────────────────────────────────────────────
Upsampling_rate = 120              # 上采样倍数
rho             = 0.4              # 根升余弦滚降系数
RC              = rcosdesign(rho, 4, Upsampling_rate)
fc              = 1.6e9            # 射频载波频率 1.6 GHz
fs              = 1e9 * Upsampling_rate   # 采样率 120 GHz

# ── 导频参数（来自 sim4，Nps=4；这里改为 Nps=4 以增大导频密度适配5径信道）──
Nps           = 4                  # 导频间隔：每 4 个子载波 1 个导频
pilot_loc_base = list(range(0, numSubcarriers, Nps))
Np            = len(pilot_loc_base)                    # 导频数 = 16
data_loc_base = [k for k in range(numSubcarriers)
                 if k not in pilot_loc_base]
Nd            = len(data_loc_base)                     # 数据子载波数 = 48

# ── 5径信道参数（来自 sim4）──────────────────────────────────────────────────
PowerdB       = [0, -8, -17, -21, -25]    # 各径功率 [dB]
Delay         = [0,  3,   5,   6,   8]    # 各径时延 [采样]
channel_length = Delay[-1] + 1            # 信道长度 = 9（用于 DFT 截断）

# ── QAM 映射表（来自 OFDM，保持一致）────────────────────────────────────────
mapping_table = {
    (0,0,0,0):-3-3j,(0,0,0,1):-3-1j,(0,0,1,0):-3+3j,(0,0,1,1):-3+1j,
    (0,1,0,0):-1-3j,(0,1,0,1):-1-1j,(0,1,1,0):-1+3j,(0,1,1,1):-1+1j,
    (1,0,0,0): 3-3j,(1,0,0,1): 3-1j,(1,0,1,0): 3+3j,(1,0,1,1): 3+1j,
    (1,1,0,0): 1-3j,(1,1,0,1): 1-1j,(1,1,1,0): 1+3j,(1,1,1,1): 1+1j,
}
avg_pwr       = np.mean([np.abs(v)**2 for v in mapping_table.values()])
scale_factor  = np.sqrt(avg_pwr)
constellation = np.array(list(mapping_table.values())) / scale_factor
bit_map       = list(mapping_table.keys())

# ── Eb/N0 扫描（x轴来自 OFDM）───────────────────────────────────────────────
EbN0_db    = np.arange(0, 30, 2)
# SNR（子载波级）= Eb/N0 + 10log10(k_bits)
# 导频开销修正：有效数据子载波 Nd/numSubcarriers，SNR 需补偿导频开销
# 与 sim4 方式三保持一致：esn0 = ebn0 + 10log10(Nd/numSubcarriers) + 10log10(ml)
pilot_overhead_dB = 10*np.log10(Nd / numSubcarriers)   # 导频开销损失（负值）
SNR_db_arr = EbN0_db + 10*np.log10(k_bits) + pilot_overhead_dB

numOFDMBlocks = 300    # 每个 SNR 点的块数（越多越平滑）

# ── 结果数组 ──────────────────────────────────────────────────────────────────
# BER：4列 = [完美均衡, LS线性, LS样条, MMSE]
# MSE：6列 = [LS线性, LS样条, MMSE, LS线性+DFT, LS样条+DFT, MMSE+DFT]
BER_all = np.zeros((len(EbN0_db), 4))
MSE_all = np.zeros((len(EbN0_db), 6))

# ── 可视化数据（取中间某个 SNR 点，更有代表性）───────────────────────────────
vis_snr_idx   = len(EbN0_db) // 2    # 取中间 SNR 点（约 14 dB）的首块
vis_done      = False
vis_H_true_dB = None
vis_H_ests_dB = {}
vis_H_dft_dB  = {}
vis_syms_pre  = []    # 均衡前（含信道畸变）
vis_syms_post = []    # 完美均衡后

print(f"5径信道参数: PowerdB={PowerdB}, Delay={Delay}")
print(f"导频: Nps={Nps}, Np={Np}/{numSubcarriers}, 数据子载波={Nd}")
print(f"\n{'Eb/N0':>6} | {'完美均衡':>10} | {'LS线性':>10} | {'LS样条':>10} | {'MMSE':>10}")
print("-" * 62)

np.random.seed(42)


# ════════════════════════════════════════════════════════════════════════════
# 主仿真循环
# ════════════════════════════════════════════════════════════════════════════

for jter, snr_db in enumerate(SNR_db_arr):
    snr_lin = 10**(snr_db / 10)

    total_errors = np.zeros(4, dtype=int)
    total_bits   = 0
    mse_sum      = np.zeros(6)

    for blk in range(numOFDMBlocks):

        # ── 导频符号（全 +1，与 sim4 一致；也可改为随机 BPSK）──────────────
        Xp = np.ones(Np, dtype=float)

        # ── 生成数据比特并 16-QAM 调制 ────────────────────────────────────
        bits = np.random.randint(0, 2, Nd * k_bits)
        data_syms = np.array([
            mapping_table[tuple(bits[i:i+k_bits])] / scale_factor
            for i in range(0, Nd*k_bits, k_bits)
        ])

        # ── 子载波复用：导频插入（与 sim4 的插入方式对应）─────────────────
        X = np.zeros(numSubcarriers, dtype=complex)
        X[pilot_loc_base] = Xp
        X[data_loc_base]  = data_syms

        # ── IFFT + 循环前缀（来自 OFDM）──────────────────────────────────
        x_time = np.fft.ifft(X) * np.sqrt(numSubcarriers)
        x_cp   = np.concatenate([x_time[-cp_len:], x_time])   # (80,)

        # ── 上采样 → RRC 脉冲成型（来自 OFDM）────────────────────────────
        N_cp  = len(x_cp)
        x_up  = np.zeros(N_cp * Upsampling_rate, dtype=complex)
        x_up[::Upsampling_rate] = x_cp
        x_rrc = np.convolve(x_up, RC, mode='full')
        Lsig  = len(x_rrc)

        # ── 射频上变频（来自 OFDM）────────────────────────────────────────
        tt    = np.arange(Lsig) / fs
        car   = np.sqrt(2) * np.exp(1j * 2*np.pi * fc * tt)
        tx_rf = np.real(x_rrc * car)              # 实数射频信号

        # ── 5径瑞利信道（来自 sim4，每块独立生成）────────────────────────
        h      = make_channel(PowerdB, Delay)
        H_true = np.fft.fft(h, numSubcarriers)   # 真实频率响应（用于完美均衡和MSE）

        # 基带多径卷积（在基带做等效，避免射频卷积开销）
        x_ch     = fftconvolve(x_cp, h)           # 长度 = N_cp + len(h) - 1
        Lch      = len(x_ch)

        # 重新上采样 + RRC 成型 + 射频上变频
        x_ch_up  = np.zeros(Lch * Upsampling_rate, dtype=complex)
        x_ch_up[::Upsampling_rate] = x_ch
        x_ch_rrc = fftconvolve(x_ch_up, RC)
        Lrf      = len(x_ch_rrc)
        tt2      = np.arange(Lrf) / fs
        car2     = np.sqrt(2) * np.exp(1j * 2*np.pi * fc * tt2)
        tx_rf2   = np.real(x_ch_rrc * car2)

        # ── AWGN 噪声（噪声功率从射频信号测量，与 OFDM 一致）────────────
        sig_pwr   = np.mean(tx_rf2**2) * Upsampling_rate * 0.5
        noise_std = np.sqrt(sig_pwr / snr_lin)
        rx_rf     = tx_rf2 + np.random.normal(0, noise_std, Lrf)

        # ── 射频下变频（来自 OFDM）────────────────────────────────────────
        rx_bb = rx_rf * np.real(car2) + 1j * rx_rf * (-np.imag(car2))

        # ── 接收 RRC 匹配滤波 + 降采样（来自 OFDM）───────────────────────
        rx_rrc  = fftconvolve(rx_bb, RC)
        start   = len(RC) - 1
        rx_down = rx_rrc[start : start + Lch*Upsampling_rate : Upsampling_rate]

        # ── 去循环前缀 → FFT（来自 OFDM）─────────────────────────────────
        # 多径使符号延长，取 [cp_len : cp_len+numSubcarriers]
        rx_nocp = rx_down[cp_len : cp_len + numSubcarriers]
        Y       = np.fft.fft(rx_nocp) / np.sqrt(numSubcarriers)

        # ── 三种信道估计（来自 matlab_ref / sim4）─────────────────────────
        H_ls_lin = ls_estimate(Y, Xp, pilot_loc_base, numSubcarriers, 'linear')
        H_ls_spl = ls_estimate(Y, Xp, pilot_loc_base, numSubcarriers, 'cubic')
        H_mmse   = mmse_estimate(
            Y, Xp, pilot_loc_base, numSubcarriers, Nps, h, snr_lin)

        # ── DFT 降噪（channel_length=9，截断保留5径有效分量）─────────────
        H_ls_lin_dft = dft_denoise(H_ls_lin, channel_length)
        H_ls_spl_dft = dft_denoise(H_ls_spl, channel_length)
        H_mmse_dft   = dft_denoise(H_mmse,   channel_length)

        # ── MSE 统计（与真实 H_true 对比）────────────────────────────────
        mse_sum[0] += np.mean(np.abs(H_true - H_ls_lin    )**2)
        mse_sum[1] += np.mean(np.abs(H_true - H_ls_spl    )**2)
        mse_sum[2] += np.mean(np.abs(H_true - H_mmse      )**2)
        mse_sum[3] += np.mean(np.abs(H_true - H_ls_lin_dft)**2)
        mse_sum[4] += np.mean(np.abs(H_true - H_ls_spl_dft)**2)
        mse_sum[5] += np.mean(np.abs(H_true - H_mmse_dft  )**2)

        # ── 均衡 + 解调（4种方法）────────────────────────────────────────
        # 完美均衡：直接用真实 H_true（对应 sim4 的 ry_per_temp）
        for mi, H_eq in enumerate([H_true, H_ls_lin, H_ls_spl, H_mmse]):
            Y_eq    = Y / H_eq                         # ZF 均衡
            rx_data = Y_eq[data_loc_base]              # 提取数据子载波

            # 最近邻判决解调
            demod_bits = []
            for sym in rx_data:
                demod_bits.extend(
                    bit_map[np.argmin(np.abs(sym - constellation))])
            total_errors[mi] += np.sum(np.array(demod_bits) != bits)

        total_bits += Nd * k_bits

        # ── 保存可视化数据（取 vis_snr_idx 对应 SNR 点的首块）────────────
        if not vis_done and jter == vis_snr_idx and blk == 0:
            vis_H_true_dB = 20*np.log10(np.abs(H_true) + 1e-12)
            for label, H_e, H_d in [
                ('LS线性', H_ls_lin, H_ls_lin_dft),
                ('LS样条', H_ls_spl, H_ls_spl_dft),
                ('MMSE',   H_mmse,   H_mmse_dft),
            ]:
                vis_H_ests_dB[label] = 20*np.log10(np.abs(H_e) + 1e-12)
                vis_H_dft_dB[label]  = 20*np.log10(np.abs(H_d) + 1e-12)
            vis_syms_pre  = Y[data_loc_base].tolist()
            vis_syms_post = (Y / H_true)[data_loc_base].tolist()
            vis_done = True

    BER_all[jter, :] = total_errors / total_bits
    MSE_all[jter, :] = mse_sum / numOFDMBlocks

    print(f"{EbN0_db[jter]:>6} | "
          f"{BER_all[jter,0]:>10.6f} | {BER_all[jter,1]:>10.6f} | "
          f"{BER_all[jter,2]:>10.6f} | {BER_all[jter,3]:>10.6f}")


# ════════════════════════════════════════════════════════════════════════════
# 绘图
# ════════════════════════════════════════════════════════════════════════════

ideal_pts = np.array(list(mapping_table.values())) / scale_factor
k_ax      = np.arange(numSubcarriers)
vis_EbN0  = EbN0_db[vis_snr_idx]

# ── 图1：BER vs Eb/N0 ─────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 6))

ber_cfg = [
    ('完美均衡（上帝视角）', '#1565C0', 'D', '-'),
    ('LS 线性插值',          '#2E7D32', 's', '--'),
    ('LS 样条插值',          '#E65100', '^', '--'),
    ('MMSE 估计',            '#B71C1C', 'o', '--'),
]
for mi, (label, col, mk, ls) in enumerate(ber_cfg):
    ax1.semilogy(EbN0_db, np.maximum(BER_all[:, mi], 1e-6),
                 color=col, marker=mk, ls=ls, lw=1.8, ms=7, label=label)

ax1.set_xlabel('Eb/N0 (dB)', fontsize=12)
ax1.set_ylabel('误码率 BER', fontsize=12)
ax1.set_title(
    '16-QAM OFDM 误码率曲线\n'
    f'5径瑞利信道  PowerdB={PowerdB}\n'
    f'Delay={Delay}  Nps={Nps}  {Np}/{numSubcarriers} 个导频  RRC + 射频链路',
    fontsize=10
)
ax1.legend(fontsize=10, loc='lower left')
ax1.grid(True, which='both', alpha=0.35)
fig1.tight_layout()


# ── 图2：MSE vs Eb/N0（6条）─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 6))

mse_cfg = [
    ('LS线性',        '#43A047', 's', '-'),
    ('LS样条',        '#FF9800', '^', '-'),
    ('MMSE',          '#E53935', 'o', '-'),
    ('LS线性 + DFT',  '#A5D6A7', 's', '--'),
    ('LS样条 + DFT',  '#FFCC80', '^', '--'),
    ('MMSE + DFT',    '#EF9A9A', 'o', '--'),
]
for mi, (label, col, mk, ls) in enumerate(mse_cfg):
    ax2.semilogy(EbN0_db, np.maximum(MSE_all[:, mi], 1e-10),
                 color=col, marker=mk, ls=ls, lw=1.8, ms=7, label=label)

ax2.set_xlabel('Eb/N0 (dB)', fontsize=12)
ax2.set_ylabel('均方误差 MSE', fontsize=12)
ax2.set_title(
    '信道估计 MSE vs Eb/N0\n'
    '实线：直接估计    虚线：+ DFT降噪',
    fontsize=11
)
ax2.legend(fontsize=9, loc='upper right', ncol=2)
ax2.grid(True, which='both', alpha=0.35)
fig2.tight_layout()


# ── 图3：信道频率响应对比（3×2子图）─────────────────────────────────────
fig3, axes3 = plt.subplots(3, 2, figsize=(14, 11))
fig3.suptitle(
    f'信道频率响应估计对比  (Eb/N0={vis_EbN0} dB，5径瑞利信道)\n'
    '左列：直接估计    右列：+ DFT降噪',
    fontsize=11, fontweight='bold'
)

for row, label in enumerate(['LS线性', 'LS样条', 'MMSE']):
    H_est_dB = vis_H_ests_dB[label]
    H_dft_dB = vis_H_dft_dB[label]

    # 左列：直接估计
    ax_l = axes3[row, 0]
    ax_l.plot(k_ax, vis_H_true_dB, 'b', lw=2,   label='真实信道 H')
    ax_l.plot(k_ax, H_est_dB,      'r--', lw=1.5, label=label)
    ax_l.scatter(pilot_loc_base,
                 [H_est_dB[p] for p in pilot_loc_base],
                 color='k', s=30, zorder=5, label='导频估计点')
    ax_l.set_title(label, fontsize=10, fontweight='bold')
    ax_l.set_xlabel('子载波序号')
    ax_l.set_ylabel('|H[k]| (dB)')
    ax_l.legend(fontsize=8); ax_l.grid(True, alpha=0.3)

    # 右列：DFT降噪后
    ax_r = axes3[row, 1]
    ax_r.plot(k_ax, vis_H_true_dB, 'b', lw=2,   label='真实信道 H')
    ax_r.plot(k_ax, H_dft_dB,      'g--', lw=1.5, label=f'{label} + DFT降噪')
    ax_r.set_title(f'{label} + DFT降噪', fontsize=10, fontweight='bold')
    ax_r.set_xlabel('子载波序号')
    ax_r.set_ylabel('|H[k]| (dB)')
    ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.3)

fig3.tight_layout()


# ── 图4：均衡前后星座图 ────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(10, 5))
fig4.suptitle(
    f'星座图  (Eb/N0={vis_EbN0} dB，5径瑞利信道)',
    fontsize=12, fontweight='bold'
)

sb = np.array(vis_syms_pre)
sa = np.array(vis_syms_post)

axes4[0].scatter(sb.real, sb.imag, s=6, color='#607D8B', alpha=0.5)
axes4[0].scatter(ideal_pts.real, ideal_pts.imag,
                 s=80, color='r', marker='+', lw=2, zorder=5)
axes4[0].set_title('均衡前（含5径信道畸变）', fontsize=10, fontweight='bold')
axes4[0].set_xlim(-4, 4); axes4[0].set_ylim(-4, 4)
axes4[0].set_aspect('equal'); axes4[0].grid(True, alpha=0.3)
axes4[0].set_xlabel('同相分量 I'); axes4[0].set_ylabel('正交分量 Q')

axes4[1].scatter(sa.real, sa.imag, s=6, color='#1565C0', alpha=0.5)
axes4[1].scatter(ideal_pts.real, ideal_pts.imag,
                 s=80, color='k', marker='+', lw=2, zorder=5)
axes4[1].set_title('完美均衡后', fontsize=10, fontweight='bold')
axes4[1].set_xlim(-2, 2); axes4[1].set_ylim(-2, 2)
axes4[1].set_aspect('equal'); axes4[1].grid(True, alpha=0.3)
axes4[1].set_xlabel('同相分量 I'); axes4[1].set_ylabel('正交分量 Q')

fig4.tight_layout()

plt.show()