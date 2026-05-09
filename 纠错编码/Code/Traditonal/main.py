"""
main.py - 主程序
功能：
    1. 验证 LDPC 和 Polar 码的正确性
    2. 打印编解码算法的复杂度分析
    3. 仿真三类实验：BER vs SNR / 码长 / 码率
    4. 生成图表（English labels，避免WSL字体问题）
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 加载本地模块
sys.path.insert(0, os.path.dirname(__file__))
from 纠错编码.Code.Traditonal.LDPC import make_ldpc_matrices, ldpc_encode, ldpc_decode, validate_ldpc
from 纠错编码.Code.Traditonal.Polar import get_frozen_and_info, polar_encode, polar_decode, validate_polar

plt.rcParams['axes.unicode_minus'] = False


# ===== 信道仿真 =====

def awgn_llr(codeword, snr_db):
    """
    AWGN信道仿真，返回信道LLR。

    调制方式：BPSK（0 → +1，1 → -1）
    信道模型：y = x + n，n ~ N(0, σ²)
    LLR公式：L = 2y / σ²

    参数：
        codeword : 发送码字，形状 (n,)，值为0/1
        snr_db   : 信噪比 Eb/N0（dB）
    返回：
        llr : 对数似然比，形状 (n,)
    """
    # BPSK调制
    x = 1 - 2 * codeword.astype(float)  # 0→+1，1→-1

    # 噪声功率
    snr_linear = 10 ** (snr_db / 10.0)
    sigma2 = 1.0 / snr_linear

    # 高斯白噪声
    noise = np.random.normal(0, np.sqrt(sigma2), len(codeword))
    y = x + noise

    # 信道LLR：正值倾向0，负值倾向1
    return 2.0 * y / sigma2


# ===== BER仿真函数 =====

def simulate_ber_ldpc(n, k, snr_db, num_frames=50):
    """
    仿真单个参数组合下LDPC码的误比特率（BER）。

    参数：
        n          : 码长
        k          : 信息位长度
        snr_db     : 信噪比（dB）
        num_frames : 仿真帧数
    返回：
        ber : 误比特率
    """
    m = n - k
    G, H = make_ldpc_matrices(k, m)

    total_bits = 0
    error_bits = 0

    for _ in range(num_frames):
        msg = np.random.randint(0, 2, k)
        codeword = ldpc_encode(msg, G)
        llr = awgn_llr(codeword, snr_db)
        decoded_codeword = ldpc_decode(llr, H, max_iter=10)
        decoded_msg = decoded_codeword[:k]

        error_bits += int(np.sum(msg != decoded_msg))
        total_bits += k

    return error_bits / total_bits if total_bits > 0 else 0.0


def simulate_ber_polar(N, K, snr_db, num_frames=50):
    """
    仿真单个参数组合下Polar码的误比特率（BER）。

    参数：
        N          : 码长（2的幂）
        K          : 信息位长度
        snr_db     : 信噪比（dB）
        num_frames : 仿真帧数
    返回：
        ber : 误比特率
    """
    info_pos, frozen_set = get_frozen_and_info(N, K)

    total_bits = 0
    error_bits = 0

    for _ in range(num_frames):
        msg = np.random.randint(0, 2, K)
        codeword = polar_encode(msg, N, info_pos)
        llr = awgn_llr(codeword, snr_db)
        decoded_msg = polar_decode(llr, N, frozen_set, info_pos)

        error_bits += int(np.sum(msg != decoded_msg))
        total_bits += K

    return error_bits / total_bits if total_bits > 0 else 0.0


# ===== 复杂度分析 =====

def print_complexity():
    """
    打印LDPC和Polar码编解码算法的复杂度分析。
    分析项：
        - 编码算法
        - 解码算法（LDPC MinSum、Polar SC）
        - 参数对复杂度的影响
    """
    print()
    print("=" * 60)
    print("编解码算法复杂度分析")
    print("=" * 60)

    print("\n【LDPC码】")
    print("-" * 60)
    print("编码（矩阵乘法 c = m·G）：")
    print("  复杂度：O(k·n)")
    print("  k=128, n=256 → 约 32,768 次运算")
    print()
    print("MinSum解码：")
    print("  E = 校验矩阵H中1的个数（总边数），E ≈ n·d_v（d_v为列重）")
    print("  每次迭代：O(E)  [校验节点更新 + 变量节点更新]")
    print("  总复杂度：O(max_iter · E)")
    print("  k=128, n=256, d_v=3, iter=10 → 约 7,680 次运算/帧")
    print()
    print("  优势：E远小于n²（稀疏矩阵），解码高效")
    print("  代价：迭代次数不固定，延迟不确定")

    print("\n【Polar码】")
    print("-" * 60)
    print("编码（蝶形变换 x = u·F^⊗n）：")
    print("  共 n = log₂N 层，每层 N/2 次异或")
    print("  复杂度：O(N·log N)")
    print("  N=256 → 约 1,024 次运算（比LDPC快约32倍）")
    print()
    print("SC解码（连续消除）：")
    print("  递归计算N个位的LLR，每个位平均 O(log N) 次操作")
    print("  总复杂度：O(N·log N)")
    print("  N=256 → 约 2,048 次运算")
    print()
    print("  优势：延迟固定，结构规整，易于硬件实现")
    print("  代价：短码性能弱于LDPC（可用SCL+CRC改善）")

    print("\n【两者对比（n/N=256，R=0.5）】")
    print("-" * 60)
    n, N = 256, 256
    k = K = 128

    ldpc_enc = k * n
    polar_enc = N * int(np.log2(N))
    ldpc_dec = 10 * n * 3   # iter × E（d_v=3）
    polar_dec = N * int(np.log2(N))

    print(f"  LDPC 编码：{ldpc_enc:,} 次运算")
    print(f"  Polar 编码：{polar_enc:,} 次运算  （快 {ldpc_enc//polar_enc} 倍）")
    print(f"  LDPC MinSum 解码：{ldpc_dec:,} 次运算")
    print(f"  Polar SC 解码：{polar_dec:,} 次运算  （快 {ldpc_dec//polar_dec} 倍）")
    print("=" * 60)


# ===== 绘图函数 =====

def plot_ber_vs_snr(snr_range, ber_ldpc, ber_polar, n, k, save_dir):
    """绘制 BER vs SNR 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(snr_range, ber_ldpc, 'o-', linewidth=2, markersize=6,
                color='#2878BD', label=f'LDPC (n={n}, k={k})')
    ax.semilogy(snr_range, ber_polar, 's--', linewidth=2, markersize=6,
                color='#E84646', label=f'Polar (N={n}, K={k})')

    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title(f'BER vs SNR  (n={n}, Rate={k/n:.2f})', fontsize=13, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=1e-4)

    plt.tight_layout()
    path = os.path.join(save_dir, 'ber_vs_snr.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  图表已保存：{path}")


def plot_ber_vs_length(lengths, ber_ldpc_list, ber_polar_list, snr_db, save_dir):
    """绘制 BER vs 码长 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(lengths, ber_ldpc_list, 'o-', linewidth=2, markersize=7,
                color='#2878BD', label='LDPC')
    ax.semilogy(lengths, ber_polar_list, 's--', linewidth=2, markersize=7,
                color='#E84646', label='Polar')

    ax.set_xlabel('Code Length n', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title(f'BER vs Code Length  (Rate=0.5, Eb/N0={snr_db}dB)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(lengths)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, 'ber_vs_length.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  图表已保存：{path}")


def plot_ber_vs_rate(rates, ber_ldpc_list, ber_polar_list, n, snr_db, save_dir):
    """绘制 BER vs 码率 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(rates, ber_ldpc_list, 'o-', linewidth=2, markersize=7,
                color='#2878BD', label='LDPC')
    ax.semilogy(rates, ber_polar_list, 's--', linewidth=2, markersize=7,
                color='#E84646', label='Polar')

    ax.set_xlabel('Code Rate R = k/n', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title(f'BER vs Code Rate  (n={n}, Eb/N0={snr_db}dB)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, 'ber_vs_rate.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  图表已保存：{path}")


def plot_complexity_comparison(save_dir):
    """绘制不同码长下的理论计算量对比"""
    lengths = [16, 32, 64, 128, 256, 512]
    rate = 0.5

    ldpc_enc_ops, ldpc_dec_ops = [], []
    polar_enc_ops, polar_dec_ops = [], []

    for n in lengths:
        k = int(n * rate)
        ldpc_enc_ops.append(k * n)
        ldpc_dec_ops.append(10 * n * 3)           # iter × E
        polar_enc_ops.append(n * int(np.log2(n)))
        polar_dec_ops.append(n * int(np.log2(n)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(lengths, ldpc_enc_ops, 'o-', linewidth=2, markersize=7,
             color='#2878BD', label='LDPC  O(k·n)')
    ax1.plot(lengths, polar_enc_ops, 's--', linewidth=2, markersize=7,
             color='#E84646', label='Polar  O(N·logN)')
    ax1.set_xlabel('Code Length n', fontsize=11)
    ax1.set_ylabel('Operations (approx.)', fontsize=11)
    ax1.set_title('Encoding Complexity', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    ax2.plot(lengths, ldpc_dec_ops, 'o-', linewidth=2, markersize=7,
             color='#2878BD', label='LDPC MinSum  O(iter·E)')
    ax2.plot(lengths, polar_dec_ops, 's--', linewidth=2, markersize=7,
             color='#E84646', label='Polar SC  O(N·logN)')
    ax2.set_xlabel('Code Length n', fontsize=11)
    ax2.set_ylabel('Operations (approx.)', fontsize=11)
    ax2.set_title('Decoding Complexity', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, 'complexity.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  图表已保存：{path}")


# ===== 主流程 =====

def main():
    save_dir = "/home/krito/Uestc/纠错编码/Code/Traditonal/Image"
    np.random.seed(42)   # 固定随机种子，结果可复现

    # ─────────────────────────────────────────────
    # 第一步：验证编解码正确性
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("第一步：验证编解码正确性")
    print("=" * 60)

    validate_ldpc(k=16, m=16)        # n=32，码率0.5
    validate_polar(N=32, K=16)       # 码率0.5

    # ─────────────────────────────────────────────
    # 第二步：复杂度分析
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("第二步：复杂度分析")
    print("=" * 60)

    print_complexity()

    print("\n  正在生成复杂度对比图...")
    plot_complexity_comparison(save_dir)

    # ─────────────────────────────────────────────
    # 第三步：BER vs SNR
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("第三步：BER vs 信噪比（SNR）")
    print("  码长=32，码率=0.5，每个SNR点50帧")
    print("=" * 60)

    n, k, N, K = 32, 16, 32, 16
    snr_range = list(range(0, 9))   # 0 ~ 8 dB
    frames = 50

    ber_ldpc_snr, ber_polar_snr = [], []
    for snr in snr_range:
        b_l = simulate_ber_ldpc(n, k, snr, frames)
        b_p = simulate_ber_polar(N, K, snr, frames)
        ber_ldpc_snr.append(b_l)
        ber_polar_snr.append(b_p)
        print(f"  SNR={snr:2d} dB | LDPC BER={b_l:.4f} | Polar BER={b_p:.4f}")

    plot_ber_vs_snr(snr_range, ber_ldpc_snr, ber_polar_snr, n, k, save_dir)

    # ─────────────────────────────────────────────
    # 第四步：BER vs 码长
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("第四步：BER vs 码长（码率=0.5，SNR=5dB）")
    print("  每个码长50帧")
    print("=" * 60)

    lengths = [16, 32, 64]
    snr_fixed = 5.0
    ber_ldpc_len, ber_polar_len = [], []

    for length in lengths:
        kk = length // 2
        b_l = simulate_ber_ldpc(length, kk, snr_fixed, frames)
        b_p = simulate_ber_polar(length, kk, snr_fixed, frames)
        ber_ldpc_len.append(b_l)
        ber_polar_len.append(b_p)
        print(f"  码长={length:3d} | LDPC BER={b_l:.4f} | Polar BER={b_p:.4f}")

    plot_ber_vs_length(lengths, ber_ldpc_len, ber_polar_len, snr_fixed, save_dir)

    # ─────────────────────────────────────────────
    # 第五步：BER vs 码率
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("第五步：BER vs 码率（码长=32，SNR=5dB）")
    print("  每个码率50帧")
    print("=" * 60)

    n_fixed = 32
    rate_ks = [8, 12, 16, 20, 24]    # k值，对应码率0.25~0.75
    rates = [kk / n_fixed for kk in rate_ks]
    ber_ldpc_rate, ber_polar_rate = [], []

    for kk in rate_ks:
        b_l = simulate_ber_ldpc(n_fixed, kk, snr_fixed, frames)
        b_p = simulate_ber_polar(n_fixed, kk, snr_fixed, frames)
        ber_ldpc_rate.append(b_l)
        ber_polar_rate.append(b_p)
        print(f"  码率={kk/n_fixed:.3f} (k={kk:2d}) | LDPC BER={b_l:.4f} | Polar BER={b_p:.4f}")

    plot_ber_vs_rate(rates, ber_ldpc_rate, ber_polar_rate, n_fixed, snr_fixed, save_dir)

    # ─────────────────────────────────────────────
    # 结果汇总
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("运行完毕！共生成4张图片：")
    for name in ['complexity.png', 'ber_vs_snr.png',
                 'ber_vs_length.png', 'ber_vs_rate.png']:
        print(f"  {os.path.join(save_dir, name)}")
    print("=" * 60)


if __name__ == "__main__":
    main()