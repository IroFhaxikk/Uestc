import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn


def rcosdesign(beta, span, sps):
    """生成根升余弦滤波器系数 (对应 MATLAB rcosdesign)"""
    delay = span * sps / 2
    t = np.arange(-delay, delay + 1) / sps
    # 处理 t=0 的奇点
    t[np.abs(t) < 1e-12] = 0

    # 分子分母计算
    num = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
    denom = np.pi * t * (1 - (4 * beta * t) ** 2)

    # 处理分母为0的情况
    with np.errstate(divide='ignore', invalid='ignore'):
        h = num / denom

    # 修正特殊点 (t=0 和 t = +/- 1/(4beta))
    h[t == 0] = 1.0 - beta + 4 * beta / np.pi
    if beta != 0:
        idx = np.abs(np.abs(t) - 1 / (4 * beta)) < 1e-5
        h[idx] = (beta / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))

    # 归一化能量
    h = h / np.sqrt(np.sum(h ** 2))
    return h


def bit_err(a, b):
    return np.mean(a != b)


# --- 仿真参数 ---
plot_num = 3200  # 总比特数
M = 16  # 16-QAM
k = int(np.log2(M))  # 4 bits per symbol
numSubcarriers = 64
cp_len = 16
Upsampling_rate = 120

# --- 滤波器设计 ---
rho = 0.4
Nsym = 4
RC = rcosdesign(rho, Nsym, Upsampling_rate)

fc = 1.6e9  # 1.6 GHz
fs = 1e9 * Upsampling_rate  # 120 GHz

EbN0_db = np.arange(0, 14, 2)
SNR_plot_db = EbN0_db + 10 * np.log10(k)
# 理论值仅供参考
QAM_16_theoretical = [0.14098, 0.097742, 0.058624, 0.027871, 0.0092472, 0.0017542, 0.00013866]

# --- 1. 生成发送数据 (一次性生成，循环内只做处理) ---
# np.random.seed(42)
# 确保总比特数能被 k 整除
num_total_bits = plot_num * k
orig_bits = np.random.randint(0, 2, num_total_bits)

# QAM 映射 (使用 Scipy 或自定义，这里为了匹配 MATLAB 默认行为，手动构建映射)
# MATLAB qammod 默认是 Binary encoding (非 Gray)，且 'UnitAveragePower'
# 构建 16-QAM 星座图
mapping_table = {
    (0, 0, 0, 0): -3 - 3j, (0, 0, 0, 1): -3 - 1j, (0, 0, 1, 0): -3 + 3j, (0, 0, 1, 1): -3 + 1j,
    (0, 1, 0, 0): -1 - 3j, (0, 1, 0, 1): -1 - 1j, (0, 1, 1, 0): -1 + 3j, (0, 1, 1, 1): -1 + 1j,
    (1, 0, 0, 0): 3 - 3j, (1, 0, 0, 1): 3 - 1j, (1, 0, 1, 0): 3 + 3j, (1, 0, 1, 1): 3 + 1j,
    (1, 1, 0, 0): 1 - 3j, (1, 1, 0, 1): 1 - 1j, (1, 1, 1, 0): 1 + 3j, (1, 1, 1, 1): 1 + 1j
}
# 归一化因子
avg_pwr = np.mean([np.abs(v) ** 2 for v in mapping_table.values()])
scale_factor = np.sqrt(avg_pwr)

tx_mod = []
for i in range(0, len(orig_bits), 4):
    bits = tuple(orig_bits[i:i + 4])
    tx_mod.append(mapping_table[bits] / scale_factor)
tx_mod = np.array(tx_mod)

# Reshape to (64, numBlocks) -> MATLAB 也是按列填充
numOFDMBlocks = len(tx_mod) // numSubcarriers
tx_mod = tx_mod.reshape(numSubcarriers, numOFDMBlocks, order='F')

BER_OFDM = np.zeros(len(EbN0_db))

print(f"{'Eb/N0 (dB)':<12} | {'BER':<10}")
print("-" * 25)

for jter, snr_db in enumerate(SNR_plot_db):

    # --- 发送端 ---

    # 1. IFFT (Column-wise)
    tx_OFDM = np.fft.ifft(tx_mod, axis=0) * np.sqrt(numSubcarriers)

    # 2. Add CP
    tx_OFDM_CP = np.vstack((tx_OFDM[-cp_len:, :], tx_OFDM))  # shape: (80, Blocks)

    # 3. Upsample (Block-wise!) & 4. Convolve (Block-wise!)
    # MATLAB: conv 对每一列单独做，导致每一列长度增加
    # 我们先手动上采样每一列
    rows_cp, cols = tx_OFDM_CP.shape
    tx_OFDM_UP = np.zeros((rows_cp * Upsampling_rate, cols), dtype=complex)
    tx_OFDM_UP[::Upsampling_rate, :] = tx_OFDM_CP

    # 对每一列进行卷积
    # 卷积后长度 = len(signal) + len(RC) - 1
    len_conv = tx_OFDM_UP.shape[0] + len(RC) - 1
    tx_OFDM_RC_PAPR = np.zeros((len_conv, cols), dtype=complex)

    for i in range(cols):
        tx_OFDM_RC_PAPR[:, i] = np.convolve(tx_OFDM_UP[:, i], RC, mode='full')

    # 5. RF Signal Construction (Parallel to Serial)
    # MATLAB: tx_signal_OFDM_power = tx_OFDM_RC_PAPR(:) -> 按列展平
    tx_signal_OFDM_power = tx_OFDM_RC_PAPR.flatten(order='F')

    # 6. RF Up-conversion
    tt = np.arange(len(tx_signal_OFDM_power)) / fs
    carrier = np.sqrt(2) * np.exp(1j * 2 * np.pi * fc * tt)  # 复载波

    # MATLAB: real(tx * carrier)
    # 注意: MATLAB代码里 carrier 是 cos + j*sin。
    # tx_Rf_OFDM_power_50W = real(tx_signal_OFDM_power .* carrier(1:size...))
    tx_Rf_OFDM_power_50W = np.real(tx_signal_OFDM_power * carrier)

    # --- 信道 (AWGN) ---
    SNR = 10 ** (snr_db / 10)
    x_power = np.mean(tx_Rf_OFDM_power_50W ** 2) * Upsampling_rate * 0.5
    noise_std = np.sqrt(x_power / SNR)
    noise = np.random.normal(0, noise_std, size=tx_Rf_OFDM_power_50W.shape)

    rx_RF_OFDM = tx_Rf_OFDM_power_50W + noise

    # --- 接收端 ---
    carrier_cos = np.real(carrier)
    carrier_sin = np.imag(carrier)

    rx_real = rx_RF_OFDM * carrier_cos
    rx_imag = rx_RF_OFDM * (-carrier_sin)  # 对应 MATLAB -carrier_sin
    rx_signal_OFDM = rx_real + 1j * rx_imag

    # 必须恢复成和 tx_OFDM_RC_PAPR 一样的矩阵结构
    rx_OFDM = rx_signal_OFDM.reshape(len_conv, cols, order='F')

    # 卷积后长度再次增加
    len_rx_conv = len_conv + len(RC) - 1
    rx_OFDM_RC = np.zeros((len_rx_conv, cols), dtype=complex)

    for i in range(cols):
        rx_OFDM_RC[:, i] = np.convolve(rx_OFDM[:, i], RC, mode='full')

    # 4. Synchronization / Slicing (去拖尾)
    # MATLAB: idx_RC = (length(RC):1:size...-length(RC)+1) (1-based)
    # MATLAB length(RC) 对应 Python len(RC)
    # 转换到 0-based: start = len(RC) - 1
    start_idx = len(RC) - 1
    end_idx = rx_OFDM_RC.shape[0] - len(RC) + 1

    rx_OFDM_RC_select = rx_OFDM_RC[start_idx:end_idx, :]

    # 5. Downsample
    rx_OFDM_DOWN = rx_OFDM_RC_select[::Upsampling_rate, :]

    # 确保长度正确 (有时会有浮点误差导致多一行，截断即可)
    expected_rows = numSubcarriers + cp_len
    rx_OFDM_DOWN = rx_OFDM_DOWN[:expected_rows, :]

    # 6. Remove CP & FFT
    rx_OFDM_remove_CP = rx_OFDM_DOWN[cp_len:, :]
    rx_OFDM_freq = np.fft.fft(rx_OFDM_remove_CP, axis=0)
    rx_signal = rx_OFDM_freq / np.sqrt(numSubcarriers)

    # 7. QAM Demod
    # 展平以便解调
    rx_syms = rx_signal.flatten(order='F')

    # 最小欧氏距离解调
    demod_bits = []
    # 创建反向映射表 (Complex -> Bits)
    # 这里暴力搜索最近点
    constellation = np.array(list(mapping_table.values())) / scale_factor
    bit_map = list(mapping_table.keys())

    for sym in rx_syms:
        # 找到最近的星座点索引
        dists = np.abs(sym - constellation)
        idx = np.argmin(dists)
        demod_bits.extend(bit_map[idx])

    demod_bits = np.array(demod_bits)

    # 8. Error Calc
    BER_OFDM[jter] = bit_err(demod_bits, orig_bits)
    print(f"{EbN0_db[jter]:<12} | {BER_OFDM[jter]:.6f}")

# --- CSV 生成 (保持不变) ---
wave_design = tx_Rf_OFDM_power_50W
lenawg = len(wave_design)
m1 = np.zeros(lenawg, dtype=int);
m1[:lenawg // 5] = 1
m2 = np.zeros(lenawg, dtype=int);
m2[:lenawg // 5] = 1
data_matrix = np.column_stack((wave_design, m1, m2))

with open('test1.csv', 'w') as fid:
    fid.write(f"SampleRate={120e9}\nSetConfig=true\nY1,SampleMarker1,SampleMarker2\n")
    for row in data_matrix:
        fid.write(f"{row[0]},{int(row[1])},{int(row[2])}\n")

# --- 绘图 ---
plt.figure()
plt.semilogy(EbN0_db, QAM_16_theoretical, '-ko', label='Theoretical')
plt.semilogy(EbN0_db, BER_OFDM, '--r', label='Simulated')
plt.grid(True, which="both");
plt.legend();
plt.xlabel('Eb/N0');
plt.ylabel('BER')
plt.title('16-QAM OFDM BER')
plt.show()