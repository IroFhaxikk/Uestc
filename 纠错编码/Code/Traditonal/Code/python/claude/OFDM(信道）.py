import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.interpolate import interp1d
from scipy.special import erfc


# ── 宋体字体（macOS / Windows / Linux）──────────────────────────────────
def _setup_cjk_font():
    candidates = [
        '/System/Library/Fonts/Supplemental/Songti.ttc',
        '/Library/Fonts/Songti.ttc',
        os.path.expanduser('~/Library/Fonts/Songti.ttc'),
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/SimSun.ttf',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    ]
    for p in candidates:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            prop = fm.FontProperties(fname=p)
            matplotlib.rcParams['font.family'] = [prop.get_name(), 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"[字体] 已加载宋体: {p}")
            return
    matplotlib.rcParams['axes.unicode_minus'] = False
    print("[字体] 未找到宋体，使用系统默认字体")

_setup_cjk_font()


# ════════════════════════════════════════════════════════════════════════
# § 1  RRC 滤波器
# ════════════════════════════════════════════════════════════════════════

def rcosdesign(beta, span, sps):
    delay = span * sps / 2
    t = np.arange(-delay, delay + 1) / sps
    t[np.abs(t) < 1e-12] = 0
    num   = np.sin(np.pi*t*(1-beta)) + 4*beta*t*np.cos(np.pi*t*(1+beta))
    denom = np.pi*t*(1-(4*beta*t)**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = num / denom
    h[t == 0] = 1.0 - beta + 4*beta/np.pi
    if beta != 0:
        idx = np.abs(np.abs(t)-1/(4*beta)) < 1e-5
        h[idx] = (beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))
                                    +(1-2/np.pi)*np.cos(np.pi/(4*beta)))
    return h / np.sqrt(np.sum(h**2))

def bit_err(a, b):
    return np.mean(a != b)


# ════════════════════════════════════════════════════════════════════════
# § 2  5径瑞利多径信道
# ════════════════════════════════════════════════════════════════════════
#
#  信道冲激响应  h[n] = Σ_l  h_l · δ(n - Delay[l])
#  每径复增益   h_l ~ CN(0, Power[l])，每块独立（块衰落）
#  CP 条件：    cp_len = 16 ≥ max(Delay) = 8  ✓

PowerdB = [0, -8, -17, -21, -25]
Delay   = [0,  3,   5,   6,   8]

def make_channel():
    Power  = 10**(np.array(PowerdB)/10)
    ch_len = Delay[-1] + 1          # = 9
    h      = np.zeros(ch_len, dtype=complex)
    for d, p in zip(Delay, Power):
        h[d] = (np.random.randn() + 1j*np.random.randn()) * np.sqrt(p/2)
    return h


# ════════════════════════════════════════════════════════════════════════
# § 3  系统参数
# ════════════════════════════════════════════════════════════════════════

M               = 16
k               = int(np.log2(M))       # 4 bits/symbol
numSubcarriers  = 64                     # N = 64
cp_len          = 16                     # CP ≥ max(Delay) = 8 ✓
Upsampling_rate = 120
rho, Nsym       = 0.4, 4
RC              = rcosdesign(rho, Nsym, Upsampling_rate)
fc              = 1.6e9
fs              = 1e9 * Upsampling_rate

# ── 导频设计（梳状，每 NPS 个子载波插 1 个导频）────────────────────────
#
#  梳状导频示意（NPS=8，共8个导频，56个数据子载波）：
#  子载波: 0  1  2  3  4  5  6  7  8  9 ...
#  类型:   P  D  D  D  D  D  D  D  P  D ...  (P=导频 D=数据)
#
NPS    = 4                                      # 导频间隔
P_LOC  = list(range(0, numSubcarriers, NPS))    # 导频下标 [0,8,16,24,32,40,48,56]
D_LOC  = [k_ for k_ in range(numSubcarriers) if k_ not in P_LOC]  # 数据下标
NP     = len(P_LOC)    # 8 个导频
ND     = len(D_LOC)    # 56 个数据子载波
Xp     = np.ones(NP)   # 导频符号全为 +1（已知，收发两端共享）

# 导频开销修正：数据子载波只有 56/64，等效 SNR 损失
pilot_overhead_dB = 10*np.log10(ND / numSubcarriers)   # ≈ -0.58 dB

EbN0_db     = np.arange(0, 30, 2)
# SNR = Eb/N0 + 10log10(k) + 导频开销修正
SNR_plot_db = EbN0_db + 10*np.log10(k) + pilot_overhead_dB

# ── 16-QAM 映射表 ────────────────────────────────────────────────────
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


# ════════════════════════════════════════════════════════════════════════
# § 4  信道估计函数
# ════════════════════════════════════════════════════════════════════════

def ce_mmse(Y, p_loc, N, Nps, h_true, snr_lin):
    """
    MMSE 信道估计（Wiener 滤波）。

    原理：
      导频处直接得到含噪声的粗估计 H̃[p] = Y[p] / Xp
      利用信道二阶统计量构建最优线性估计器：
        H_mmse = Rhp · (Rpp + I/SNR)^{-1} · H̃_p

    相关矩阵（指数功率延迟谱模型）：
      Rhp[k,p] = 1 / (1 + j·2π·τ_rms·(k - p·Nps)/N)
      Rpp[p,q] = 1 / (1 + j·2π·τ_rms·(p-q)·Nps/N)  + δ(p-q)/SNR

    τ_rms：RMS 时延扩展，由真实 h 计算（实际系统可用预设模型值）
    """
    p_loc   = np.array(p_loc)
    Np      = len(p_loc)
    H_tilde = Y[p_loc] / Xp          # 导频处 LS 初估（含噪声）

    # ── 计算 RMS 时延扩展 ────────────────────────────────────────────
    k_h     = np.arange(len(h_true), dtype=float)
    hh      = np.dot(h_true, h_true.conj()).real
    tmp     = h_true * h_true.conj() * k_h
    r1      = np.sum(tmp).real / hh
    r2      = np.dot(tmp, k_h).real / hh
    tau_rms = np.sqrt(max(r2 - r1**2, 0))

    # ── 构建相关矩阵 ─────────────────────────────────────────────────
    j2pi = 1j * 2 * np.pi * tau_rms / N

    K1  = np.tile(np.arange(N)[:, None],  (1, Np))   # (N, Np)
    K2  = np.tile(np.arange(Np)[None, :], (N, 1))    # (N, Np)
    Rhp = 1.0 / (1 + j2pi * (K1 - K2 * Nps))         # 互相关矩阵

    K3  = np.tile(np.arange(Np)[:, None], (1, Np))   # (Np, Np)
    K4  = np.tile(np.arange(Np)[None, :], (Np, 1))   # (Np, Np)
    Rpp = 1.0 / (1 + j2pi * Nps * (K3 - K4)) + np.eye(Np) / snr_lin

    return Rhp @ np.linalg.solve(Rpp, H_tilde)       # (N,)

def dft_denoise(H_est, ch_len):
    """
    DFT 降噪：IFFT → 截断前 ch_len 个时域抽头 → FFT。
    噪声在时域均匀分布在 N 个位置，截断后压缩 N/ch_len 倍。
    """
    h_t = np.fft.ifft(H_est)
    return np.fft.fft(h_t[:ch_len], len(H_est))


# ════════════════════════════════════════════════════════════════════════
# § 5  信道均衡函数
# ════════════════════════════════════════════════════════════════════════

def eq_mmse(Y, H, snr_lin):
    """
    MMSE 频域均衡。

    原理：Y[k] = H[k]·X[k] + N[k]
    最优权向量：Q[k] = H*[k] / (|H[k]|² + 1/SNR)

    深衰落子载波（|H[k]|≈0）：分母由 1/SNR 主导，不会无限放大噪声
    高 SNR 时：1/SNR → 0，退化为 ZF（直接除以 H[k]）
    """
    return Y * H.conj() / (np.abs(H)**2 + 1.0/snr_lin)


# ════════════════════════════════════════════════════════════════════════
# § 6  生成发送数据（ND 个数据子载波 × numOFDMBlocks 块）
# ════════════════════════════════════════════════════════════════════════

plot_num       = 3200
num_total_bits = plot_num * k        # 总比特数（针对数据子载波）
orig_bits      = np.random.randint(0, 2, num_total_bits)

# 每块 ND=56 个数据符号，共 plot_num//ND 块
tx_data_syms = np.array([
    mapping_table[tuple(orig_bits[i:i+4])] / scale_factor
    for i in range(0, len(orig_bits), 4)
])
numOFDMBlocks  = len(tx_data_syms) // ND
tx_data_syms   = tx_data_syms[:numOFDMBlocks * ND].reshape(ND, numOFDMBlocks, order='F')


# ════════════════════════════════════════════════════════════════════════
# § 7  主仿真循环
# ════════════════════════════════════════════════════════════════════════

ch_len     = Delay[-1] + 1          # 信道冲激响应长度 = 9
conv_len_b = (numSubcarriers + cp_len) + ch_len - 1   # 多径卷积后长度 = 88

BER_AWGN   = np.zeros(len(EbN0_db))   # AWGN 基准（无多径，无均衡）
BER_noEQ   = np.zeros(len(EbN0_db))   # 5径多径，无均衡
BER_MMSE   = np.zeros(len(EbN0_db))   # 5径多径，MMSE估计+均衡
MSE_CE     = np.zeros(len(EbN0_db))   # 信道估计 MSE

# 可视化数据（取最高 SNR 点，收集所有块）
vis_idx  = len(EbN0_db) - 1   # 取最高 SNR 点（28 dB），星座最清晰
vis_done = False
vis_data = {}
vis_Y_before_all = []   # 均衡前所有块数据子载波
vis_Y_after_all  = []   # 均衡后所有块数据子载波

print("=" * 65)
print("  16-QAM OFDM  |  5径瑞利多径信道  |  MMSE估计 + MMSE均衡")
print(f"  PowerdB={PowerdB}")
print(f"  Delay  ={Delay}   ch_len={ch_len}")
print(f"  CP={cp_len} >= max(Delay)={max(Delay)} ✓")
print(f"  导频 Nps={NPS}，{NP}导频/{numSubcarriers}子载波，{ND}数据子载波")
print("=" * 65)
print(f"\n{'Eb/N0':>5} | {'AWGN':>8} | {'多径无均衡':>10} | {'MMSE估+均':>10} | {'MSE':>10}")
print("-" * 55)

np.random.seed(42)

for jter, snr_db in enumerate(SNR_plot_db):

    ebn0    = EbN0_db[jter]
    snr_lin = 10**(snr_db / 10)
    cols    = numOFDMBlocks

    rows_cp = numSubcarriers + cp_len   # = 80

    # ── 【发射端】────────────────────────────────────────────────────

    # 子载波复用：每块插入导频（梳状）+ 数据
    X_tx = np.zeros((numSubcarriers, cols), dtype=complex)
    X_tx[P_LOC, :] = Xp[:, None]           # 导频列：每列均为 +1
    X_tx[D_LOC, :] = tx_data_syms          # 数据列

    # IFFT + CP
    tx_OFDM    = np.fft.ifft(X_tx, axis=0) * np.sqrt(numSubcarriers)
    tx_OFDM_CP = np.vstack([tx_OFDM[-cp_len:, :], tx_OFDM])  # (80, cols)

    # 上采样 + RRC + 并串 + 射频上变频
    tx_UP = np.zeros((rows_cp * Upsampling_rate, cols), dtype=complex)
    tx_UP[::Upsampling_rate, :] = tx_OFDM_CP
    len_conv = tx_UP.shape[0] + len(RC) - 1
    tx_RC    = np.zeros((len_conv, cols), dtype=complex)
    for i in range(cols):
        tx_RC[:, i] = np.convolve(tx_UP[:, i], RC, mode='full')

    tx_sig  = tx_RC.flatten(order='F')
    tt      = np.arange(len(tx_sig)) / fs
    carrier = np.sqrt(2) * np.exp(1j * 2*np.pi * fc * tt)
    tx_RF   = np.real(tx_sig * carrier)

    # ── 【信道 A：纯 AWGN】───────────────────────────────────────────

    x_pwr_a   = np.mean(tx_RF**2) * Upsampling_rate * 0.5
    rx_RF_awgn = tx_RF + np.random.normal(0, np.sqrt(x_pwr_a/snr_lin), len(tx_RF))

    # ── 【信道 B：5径瑞利多径 + AWGN】───────────────────────────────
    #
    # 逐块卷积：每块用独立信道（块衰落），卷积后截取前 N+CP 个采样
    # 多径拖尾（ch_len-1=8 个采样）被 CP 吸收，不污染 FFT 窗口

    tx_CH = np.zeros((rows_cp, cols), dtype=complex)
    h_blocks = []   # 保存每块的信道，供接收端 MMSE 使用

    for blk in range(cols):
        h = make_channel()
        h_blocks.append(h)
        x_ch = np.convolve(tx_OFDM_CP[:, blk], h, mode='full')  # 长度 88
        tx_CH[:, blk] = x_ch[:rows_cp]    # 截取前 80 个（拖尾落在 CP 内）

    # 上采样 + RRC + 并串 + 射频上变频
    tx_CH_UP = np.zeros((rows_cp * Upsampling_rate, cols), dtype=complex)
    tx_CH_UP[::Upsampling_rate, :] = tx_CH
    len_conv_ch = tx_CH_UP.shape[0] + len(RC) - 1
    tx_CH_RC    = np.zeros((len_conv_ch, cols), dtype=complex)
    for i in range(cols):
        tx_CH_RC[:, i] = np.convolve(tx_CH_UP[:, i], RC, mode='full')

    tx_sig_ch  = tx_CH_RC.flatten(order='F')
    tt_ch      = np.arange(len(tx_sig_ch)) / fs
    carrier_ch = np.sqrt(2) * np.exp(1j * 2*np.pi * fc * tt_ch)
    tx_RF_ch   = np.real(tx_sig_ch * carrier_ch)

    x_pwr_ch  = np.mean(tx_RF_ch**2) * Upsampling_rate * 0.5
    rx_RF_ch  = tx_RF_ch + np.random.normal(0, np.sqrt(x_pwr_ch/snr_lin), len(tx_RF_ch))

    # ── 【接收端公共子函数】──────────────────────────────────────────

    def rx_frontend(rx_RF, car, lc, nc):
        """下变频 → RRC → 降采样 → 去CP → FFT，返回频域矩阵 (N, nc)"""
        rx_bb  = rx_RF * np.real(car) + 1j*rx_RF*(-np.imag(car))
        rx_mat = rx_bb.reshape(lc, nc, order='F')
        lr     = lc + len(RC) - 1
        rx_RC  = np.zeros((lr, nc), dtype=complex)
        for i in range(nc):
            rx_RC[:, i] = np.convolve(rx_mat[:, i], RC, mode='full')
        s   = len(RC) - 1
        e   = rx_RC.shape[0] - len(RC) + 1
        rdn = rx_RC[s:e:Upsampling_rate, :][:numSubcarriers+cp_len, :]
        Y_freq = np.fft.fft(rdn[cp_len:, :], axis=0) / np.sqrt(numSubcarriers)
        return Y_freq   # (N, cols)

    def demodulate(syms_flat):
        """最近邻硬判决解调"""
        bits = []
        for sym in syms_flat:
            bits.extend(bit_map[np.argmin(np.abs(sym - constellation))])
        return np.array(bits)

    # ── 【接收端 A：AWGN，无均衡（基准）】───────────────────────────

    Y_awgn   = rx_frontend(rx_RF_awgn, carrier, len_conv, cols)
    # AWGN 路径只取数据子载波（无导频，与原版保持一致）
    # 注：此处 X_tx 含导频，为使 BER 计算公平，只统计数据子载波
    rx_syms_awgn = Y_awgn[D_LOC, :].flatten(order='F')
    bits_awgn    = demodulate(rx_syms_awgn)
    BER_AWGN[jter] = bit_err(bits_awgn, orig_bits[:len(bits_awgn)])

    # ── 【接收端 B：多径，无均衡】───────────────────────────────────

    Y_mp        = rx_frontend(rx_RF_ch, carrier_ch, len_conv_ch, cols)
    rx_syms_mp  = Y_mp[D_LOC, :].flatten(order='F')
    bits_mp     = demodulate(rx_syms_mp)
    BER_noEQ[jter] = bit_err(bits_mp, orig_bits[:len(bits_mp)])

    # ── 【接收端 C：多径 + MMSE信道估计 + MMSE均衡】──────────────────
    #
    # 逐块处理：
    #   1. 导频处得到含噪估计 H̃[p] = Y[p] / Xp
    #   2. MMSE Wiener 滤波 → H_mmse (N个子载波)
    #   3. DFT 降噪（截断时域抽头）
    #   4. MMSE 均衡 → Y_eq[k] = Y[k]·H*[k]/(|H[k]|²+1/SNR)
    #   5. 取数据子载波硬判决

    bits_mmse  = []
    mse_sum    = 0.0

    for blk in range(cols):
        Y_blk  = Y_mp[:, blk]                      # (N,) 当前块频域接收
        h_true = h_blocks[blk]                      # 当前块真实信道
        H_true = np.fft.fft(h_true, numSubcarriers) # 真实频响（用于 MSE）

        # MMSE 信道估计
        H_est = ce_mmse(Y_blk, P_LOC, numSubcarriers, NPS, h_true, snr_lin)

        # DFT 降噪（截断至 ch_len=9 个时域抽头）
        H_est = dft_denoise(H_est, ch_len)

        # MSE 统计
        mse_sum += np.mean(np.abs(H_true - H_est)**2)

        # MMSE 均衡
        Y_eq = eq_mmse(Y_blk, H_est, snr_lin)

        # 提取数据子载波解调
        for sym in Y_eq[D_LOC]:
            bits_mmse.extend(bit_map[np.argmin(np.abs(sym - constellation))])

        # 保存可视化数据（最高 SNR，收集所有块）
        if jter == vis_idx:
            vis_Y_before_all.append(Y_blk[D_LOC])   # 均衡前（所有块累积）
            vis_Y_after_all.append(Y_eq[D_LOC])      # 均衡后（所有块累积）
            if blk == 0:
                vis_data['H_true_dB'] = 20*np.log10(np.abs(H_true)+1e-12)
                vis_data['H_est_dB']  = 20*np.log10(np.abs(H_est) +1e-12)
                vis_data['ebn0']      = ebn0
    # 所有块收集完后合并
    if jter == vis_idx:
        vis_data['Y_before'] = np.concatenate(vis_Y_before_all)
        vis_data['Y_after']  = np.concatenate(vis_Y_after_all)

    bits_mmse = np.array(bits_mmse)
    BER_MMSE[jter]  = bit_err(bits_mmse, orig_bits[:len(bits_mmse)])
    MSE_CE[jter]    = mse_sum / cols

    print(f"{ebn0:>5} | "
          f"{BER_AWGN[jter]:>8.5f} | "
          f"{BER_noEQ[jter]:>10.5f} | "
          f"{BER_MMSE[jter]:>10.5f} | "
          f"{MSE_CE[jter]:>10.4e}")


# ════════════════════════════════════════════════════════════════════════
# § 8  绘图
# ════════════════════════════════════════════════════════════════════════

EbN0_th  = np.linspace(0, 28, 300)
EbN0_lin = 10**(EbN0_th/10)
th_awgn  = (3/8) * erfc(np.sqrt(EbN0_lin * 4/5))
gamma    = EbN0_lin * 4/5
th_ray   = (3/8) * (1 - np.sqrt(gamma/(1+gamma)))

IDEAL_PTS = np.array(list(mapping_table.values())) / scale_factor

# ── 图1：BER vs Eb/N0（4条曲线）────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 7))

ax1.semilogy(EbN0_th, th_awgn,
             'k-',  lw=2,   label='理论：16-QAM AWGN（无编码）')
ax1.semilogy(EbN0_th, th_ray,
             'k--', lw=1.5, label='理论：16-QAM 单径瑞利（参考）')
ax1.semilogy(EbN0_db, np.maximum(BER_AWGN,  1e-6),
             'b-o', lw=1.8, ms=7, label='仿真：AWGN 信道（基准）')
ax1.semilogy(EbN0_db, np.maximum(BER_noEQ,  1e-6),
             'r-s', lw=1.8, ms=7,
             label=f'仿真：5径多径，无均衡（BER≈0.44，误码率平台）')
ax1.semilogy(EbN0_db, np.maximum(BER_MMSE,  1e-6),
             'g-^', lw=2,   ms=8,
             label='仿真：5径多径 + MMSE信道估计 + MMSE均衡')

ax1.set_xlabel('Eb/N0 (dB)', fontsize=13)
ax1.set_ylabel('误码率 BER', fontsize=13)
ax1.set_title(
    '16-QAM OFDM 误码率对比\n'
    'AWGN  vs  5径瑞利多径（无均衡）  vs  5径多径+MMSE估计+MMSE均衡\n'
    f'PowerdB={PowerdB}   Delay={Delay}   Nps={NPS}',
    fontsize=10
)
ax1.legend(fontsize=9, loc='lower left')
ax1.grid(True, which='both', alpha=0.3)
ax1.set_ylim(1e-5, 1.0)
ax1.set_xlim(EbN0_db[0]-0.5, EbN0_db[-1]+0.5)
fig1.tight_layout()

# ── 图2：MSE vs Eb/N0 ────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.semilogy(EbN0_db, np.maximum(MSE_CE, 1e-10),
             'm-o', lw=2, ms=7, label='MMSE估计（+DFT降噪）')
ax2.set_xlabel('Eb/N0 (dB)', fontsize=12)
ax2.set_ylabel('信道估计 MSE', fontsize=12)
ax2.set_title('MMSE 信道估计 MSE vs Eb/N0\n（5径瑞利信道，DFT降噪后）', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, which='both', alpha=0.3)
fig2.tight_layout()

# ── 图3：信道频率响应（可视化块）────────────────────────────────────
if vis_data:
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    fig3.suptitle(f'信道频率响应估计（Eb/N0={vis_data["ebn0"]} dB，5径瑞利信道）',
                  fontsize=12, fontweight='bold')

    # 左：频率响应对比
    ax3l = axes3[0]
    k_ax = np.arange(numSubcarriers)
    ax3l.plot(k_ax, vis_data['H_true_dB'], 'b',  lw=2,   label='真实信道 H')
    ax3l.plot(k_ax, vis_data['H_est_dB'],  'r--', lw=1.8, label='MMSE估计（+DFT降噪）')
    ax3l.scatter(P_LOC, [vis_data['H_est_dB'][p] for p in P_LOC],
                 color='k', s=40, zorder=5, label=f'导频估计点（Nps={NPS}）')
    ax3l.set_xlabel('子载波序号 k', fontsize=11)
    ax3l.set_ylabel('|H[k]| (dB)', fontsize=11)
    ax3l.set_title('频率响应（频率选择性衰落）', fontsize=11)
    ax3l.legend(fontsize=9)
    ax3l.grid(True, alpha=0.3)

    # 右：时域冲激响应（新生成一次用于展示）
    ax3r = axes3[1]
    h_show = make_channel()
    ax3r.stem(np.arange(len(h_show)), np.abs(h_show),
              linefmt='b-', markerfmt='bo', basefmt='k-')
    ax3r.axvline(cp_len, color='r', ls='--', lw=1.2, label=f'CP长度={cp_len}')
    ax3r.set_xlabel('时延（采样数）', fontsize=11)
    ax3r.set_ylabel('|h[n]|', fontsize=11)
    ax3r.set_title('5径信道冲激响应（一次实现）', fontsize=11)
    ax3r.legend(fontsize=9)
    ax3r.grid(True, alpha=0.3)

    fig3.tight_layout()

# ── 图4：星座图（三列对比：无均衡 / MMSE均衡 / 理想AWGN）────────────
if vis_data:
    fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5.5))
    ebn0_vis = vis_data["ebn0"]
    fig4.suptitle(
        f'16-QAM 星座图对比（Eb/N0={ebn0_vis} dB，5径瑞利信道，全部{numOFDMBlocks}块）',
        fontsize=12, fontweight='bold'
    )

    # 左：多径无均衡
    ax = axes4[0]
    s  = vis_data['Y_before']
    ax.scatter(s.real, s.imag, s=2, color='tomato', alpha=0.4, rasterized=True)
    ax.scatter(IDEAL_PTS.real, IDEAL_PTS.imag,
               s=80, color='k', marker='+', lw=2, zorder=5)
    ax.set_title('多径信道均衡前\n（频率选择性衰落，星座弥散）', fontsize=10, fontweight='bold')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
    ax.set_xlabel('同相分量 I'); ax.set_ylabel('正交分量 Q')
    ax.text(0.02, 0.98, f'BER≈{BER_noEQ[-1]:.3f}',
            transform=ax.transAxes, va='top', fontsize=9, color='red')

    # 中：MMSE均衡后
    ax = axes4[1]
    s  = vis_data['Y_after']
    ax.scatter(s.real, s.imag, s=2, color='steelblue', alpha=0.4, rasterized=True)
    ax.scatter(IDEAL_PTS.real, IDEAL_PTS.imag,
               s=80, color='k', marker='+', lw=2, zorder=5)
    ax.set_title('MMSE估计+均衡后\n（星座点聚拢，16点清晰可辨）', fontsize=10, fontweight='bold')
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
    ax.set_xlabel('同相分量 I'); ax.set_ylabel('正交分量 Q')
    ax.text(0.02, 0.98, f'BER≈{BER_MMSE[-1]:.4f}',
            transform=ax.transAxes, va='top', fontsize=9, color='steelblue')

    # 右：AWGN参考（理想）
    ax = axes4[2]
    s  = Y_awgn[D_LOC, :].flatten(order='F')   # 最后一轮 AWGN 接收
    ax.scatter(s.real, s.imag, s=2, color='#2E7D32', alpha=0.4, rasterized=True)
    ax.scatter(IDEAL_PTS.real, IDEAL_PTS.imag,
               s=80, color='k', marker='+', lw=2, zorder=5)
    ax.set_title('AWGN信道参考\n（无多径，无均衡，理想情况）', fontsize=10, fontweight='bold')
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
    ax.set_xlabel('同相分量 I'); ax.set_ylabel('正交分量 Q')
    ax.text(0.02, 0.98, f'BER≈{BER_AWGN[-1]:.5f}',
            transform=ax.transAxes, va='top', fontsize=9, color='#2E7D32')

    fig4.tight_layout()

plt.show()