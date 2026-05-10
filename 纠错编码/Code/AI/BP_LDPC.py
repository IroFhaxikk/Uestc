"""
neural_ldpc_analysis.py — Neural BP 智能解码器的完整仿真分析

AI 模型  ：Neural BP（加权归一化 MinSum）
编码方案 ：LDPC 码
分析内容 ：
    1. 验证 Neural BP 解码正确性
    2. 展示训练过程（损失曲线 + 权重演化）
    3. BER vs 信噪比（SNR）：Neural BP 对比传统 MinSum
    4. BER vs 码长           ：两种算法随码长的性能变化
    5. BER vs 码率           ：两种算法随码率的性能变化

图表全部使用英文标签（避免 WSL/Linux 字体乱码），
终端输出使用中文。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from LDPC import make_ldpc_matrices, ldpc_encode, ldpc_decode
from BP import (
    precompute_neighbors,
    fast_weighted_minsum,
    train_neural_bp,
    simulate_ber_neural,
    validate_neural_bp,
)

plt.rcParams['axes.unicode_minus'] = False
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(2024)   # 固定随机种子，结果可复现


# ===== 传统 MinSum 的 BER 仿真（用于对比基线）=====

def simulate_ber_minsum(H, G, snr_db, num_frames=200, max_iter=10):
    """
    传统 MinSum 解码器的误比特率仿真（权重全为 1 的 Neural BP）。
    与 Neural BP 使用完全相同的解码函数，仅权重不同，对比公平。
    """
    m, n = H.shape
    k = n - m
    weights_trad = np.ones(max_iter)          # 权重全为1 = 传统 MinSum
    row_nbrs, col_nbrs = precompute_neighbors(H)
    snr_linear = 10 ** (snr_db / 10.0)
    sigma2 = 1.0 / snr_linear

    errors, total = 0, 0
    for _ in range(num_frames):
        msg = np.random.randint(0, 2, k)
        codeword = ldpc_encode(msg, G)
        x = 1.0 - 2.0 * codeword
        noise = np.random.normal(0, np.sqrt(sigma2), n)
        llr = 2.0 * (x + noise) / sigma2
        decoded, _ = fast_weighted_minsum(
            llr, H, row_nbrs, col_nbrs, weights_trad, max_iter
        )
        errors += int(np.sum(msg != decoded[:k]))
        total += k

    return errors / total if total > 0 else 0.0


# ===== 绘图函数 =====

def plot_training_progress(loss_curve, weights, n, k, save_dir):
    """
    绘制训练过程：左图为损失曲线，右图为各轮次的学习权重。
    """
    num_iter = len(weights)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(range(1, len(loss_curve) + 1), loss_curve,
             color='#2878BD', linewidth=2)
    ax1.set_xlabel('Training Epoch', fontsize=11)
    ax1.set_ylabel('BCE Loss', fontsize=11)
    ax1.set_title(f'Training Loss Curve  (n={n}, k={k})', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 权重柱状图
    colors = ['#E84646' if w < 1.0 else '#1D9E75' for w in weights]
    bars = ax2.bar(range(1, num_iter + 1), weights, color=colors, alpha=0.85, edgecolor='white')
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Initial weight = 1.0')
    ax2.set_xlabel('Iteration Index', fontsize=11)
    ax2.set_ylabel('Learned Weight', fontsize=11)
    ax2.set_title('Learned Weights per Iteration', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(1, num_iter + 1))
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

    # 在每个柱顶标注数值
    for bar, w in zip(bars, weights):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{w:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'neural_training.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存：{path}")


def plot_ber_vs_snr(snr_range, ber_trad, ber_neural, n, k, save_dir):
    """BER vs SNR：传统 MinSum 与 Neural BP 对比"""
    def nonzero(snr_list, ber_list):
        pairs = [(s, b) for s, b in zip(snr_list, ber_list) if b > 0]
        return zip(*pairs) if pairs else ([], [])

    sl, bl = nonzero(snr_range, ber_trad)
    sn, bn = nonzero(snr_range, ber_neural)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(list(sl), list(bl), 'o-', linewidth=2, markersize=6,
                color='#2878BD', label='Traditional MinSum')
    ax.semilogy(list(sn), list(bn), 's--', linewidth=2, markersize=6,
                color='#E84646', label=f'Neural BP (weights learned)')

    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title(f'BER vs SNR — Neural BP vs Traditional MinSum\n'
                 f'LDPC (n={n}, k={k}, Rate={k/n:.2f})',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(min(snr_range) - 0.5, max(snr_range) + 0.5)
    ax.grid(True, which='both', linestyle='--', alpha=0.45)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, 'neural_ber_vs_snr.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存：{path}")


def plot_ber_vs_length(lengths, ber_trad_list, ber_neural_list, snr_db, save_dir):
    """BER vs 码长：两种算法的性能随码长变化"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # 过滤 BER=0 的点（semilogy 无法显示）
    def safe(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if y > 0]
        return zip(*pairs) if pairs else ([], [])

    xl, yl = safe(lengths, ber_trad_list)
    xn, yn = safe(lengths, ber_neural_list)

    ax.semilogy(list(xl), list(yl), 'o-', linewidth=2, markersize=8,
                color='#2878BD', label='Traditional MinSum')
    ax.semilogy(list(xn), list(yn), 's--', linewidth=2, markersize=8,
                color='#E84646', label='Neural BP')

    ax.set_xlabel('Code Length n', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title(f'BER vs Code Length  (Rate=0.5, Eb/N0={snr_db}dB)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(lengths)
    ax.grid(True, which='both', linestyle='--', alpha=0.45)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, 'neural_ber_vs_length.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存：{path}")


def plot_ber_vs_rate(rates, ber_trad_list, ber_neural_list, n, snr_db, save_dir):
    """BER vs 码率：两种算法的性能随码率变化"""
    fig, ax = plt.subplots(figsize=(8, 5))

    def safe(xs, ys):
        pairs = [(x, y) for x, y in zip(xs, ys) if y > 0]
        return zip(*pairs) if pairs else ([], [])

    xr, yr = safe(rates, ber_trad_list)
    xn, yn = safe(rates, ber_neural_list)

    ax.semilogy(list(xr), list(yr), 'o-', linewidth=2, markersize=8,
                color='#2878BD', label='Traditional MinSum')
    ax.semilogy(list(xn), list(yn), 's--', linewidth=2, markersize=8,
                color='#E84646', label='Neural BP')

    ax.set_xlabel('Code Rate R = k/n', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title(f'BER vs Code Rate  (n={n}, Eb/N0={snr_db}dB)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.45)
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, 'neural_ber_vs_rate.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存：{path}")


def plot_snr_gain(snr_range, ber_trad, ber_neural, n, k, save_dir):
    """
    绘制 Neural BP 相对于传统 MinSum 的编码增益（dB）。
    在同一 BER 下，Neural BP 需要更低的 SNR，差值即为增益。
    """
    # 在固定 BER 值下插值计算所需 SNR（仅在有效范围内）
    valid_trad = [(s, b) for s, b in zip(snr_range, ber_trad) if 1e-5 < b < 0.5]
    valid_neural = [(s, b) for s, b in zip(snr_range, ber_neural) if 1e-5 < b < 0.5]

    if len(valid_trad) < 3 or len(valid_neural) < 3:
        print("  （有效数据点不足，跳过增益图）")
        return

    ber_levels = [0.1, 0.05, 0.02, 0.01, 0.005]
    gains = []
    ber_plot = []

    for ber_target in ber_levels:
        # 在传统 MinSum 曲线上找到该 BER 对应的 SNR（插值）
        snr_trad_interp = np.interp(
            np.log10(ber_target),
            [-np.log10(b) for _, b in reversed(valid_trad)],
            [s for s, _ in reversed(valid_trad)]
        )
        snr_neural_interp = np.interp(
            np.log10(ber_target),
            [-np.log10(b) for _, b in reversed(valid_neural)],
            [s for s, _ in reversed(valid_neural)]
        )
        gain = snr_trad_interp - snr_neural_interp
        if not np.isnan(gain):
            gains.append(gain)
            ber_plot.append(ber_target)

    if not gains:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f'{b:.3f}' for b in ber_plot], gains,
           color='#E84646', alpha=0.8, edgecolor='white')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Target BER', fontsize=11)
    ax.set_ylabel('Coding Gain (dB)', fontsize=11)
    ax.set_title(f'SNR Gain of Neural BP over Traditional MinSum\n'
                 f'LDPC (n={n}, k={k})', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, 'neural_snr_gain.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存：{path}")


# ===== 主流程 =====

def main():
    t_start = time.time()

    # ────────────────────────────────────────────────
    # 第一步：验证 Neural BP 解码正确性
    # ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("第一步：验证 Neural BP 解码正确性")
    print("  使用初始权重（全1 = 传统 MinSum）验证零噪声解码")
    print("=" * 60)

    k_val, m_val = 16, 16    # n=32，码率 0.5
    G_val, H_val = make_ldpc_matrices(k_val, m_val)
    init_weights = np.ones(5)   # 5次迭代，权重全1
    validate_neural_bp(H_val, G_val, init_weights, num_iter=5)

    # ────────────────────────────────────────────────
    # 第二步：训练 Neural BP（主码 n=32, k=16）
    # ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("第二步：训练 Neural BP（n=32, k=16，混合SNR训练）")
    print("  优化目标：BCE 损失（BER 的平滑代理）")
    print("  优化算法：Adam + 数值梯度（中心差分法）")
    print("  训练SNR ：[2, 4, 6] dB 随机混合")
    print("=" * 60)

    n_main, k_main = 32, 16
    G_main, H_main = make_ldpc_matrices(k_main, n_main - k_main)
    print("\n  开始训练...\n")
    weights_main, loss_curve = train_neural_bp(
        H_main, G_main,
        num_iter=5,
        snr_db_list=[2.0, 4.0, 6.0],
        num_epochs=40,
        lr=0.08,
        batch_size=15,
        epsilon=0.06,
        verbose=True,
    )
    print(f"\n  训练完成！")
    print(f"  最终权重：{np.round(weights_main, 4)}")
    print(f"  初始权重：{np.ones(5)}")
    print(f"  权重平均偏差：{np.mean(np.abs(weights_main - 1.0)):.4f}")

    # 验证训练后的解码器
    print()
    validate_neural_bp(H_main, G_main, weights_main)

    # 绘制训练过程图
    print("\n  绘制训练过程图...")
    plot_training_progress(loss_curve, weights_main, n_main, k_main, SAVE_DIR)

    # ────────────────────────────────────────────────
    # 第三步：BER vs SNR（Neural BP vs 传统 MinSum）
    # ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("第三步：BER vs 信噪比（SNR）")
    print(f"  码长={n_main}，码率={k_main/n_main:.2f}，每个SNR点200帧")
    print("=" * 60)

    snr_range = list(range(-2, 13))    # -2 ~ 12 dB
    frames_snr = 200
    ber_trad_snr, ber_neural_snr = [], []

    print(f"\n  {'SNR(dB)':>8} | {'传统MinSum BER':>14} | {'Neural BP BER':>14} | {'增益':>8}")
    print("  " + "-" * 55)

    for snr in snr_range:
        b_t = simulate_ber_minsum(H_main, G_main, snr, frames_snr, max_iter=5)
        b_n = simulate_ber_neural(H_main, G_main, weights_main, snr, frames_snr)
        ber_trad_snr.append(b_t)
        ber_neural_snr.append(b_n)

        # 计算 BER 改善倍数
        if b_t > 0 and b_n > 0:
            gain_str = f"{b_t/b_n:.2f}×"
        elif b_t > 0 and b_n == 0:
            gain_str = "→0"
        else:
            gain_str = "both=0"
        print(f"  {snr:>8} | {b_t:>14.6f} | {b_n:>14.6f} | {gain_str:>8}")

    print("\n  绘制 BER vs SNR 图...")
    plot_ber_vs_snr(snr_range, ber_trad_snr, ber_neural_snr, n_main, k_main, SAVE_DIR)
    plot_snr_gain(snr_range, ber_trad_snr, ber_neural_snr, n_main, k_main, SAVE_DIR)

    # ────────────────────────────────────────────────
    # 第四步：BER vs 码长
    # ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("第四步：BER vs 码长（码率=0.5，SNR=5dB）")
    print("  为每种码长单独训练 Neural BP")
    print("=" * 60)

    lengths = [16, 32, 64]
    snr_len = 5.0
    frames_len = 150
    ber_trad_len, ber_neural_len = [], []

    for length in lengths:
        kk = length // 2    # 码率固定 0.5
        print(f"\n  [码长={length}, k={kk}] 训练 Neural BP...", end='', flush=True)
        G_l, H_l = make_ldpc_matrices(kk, length - kk)
        w_l, _ = train_neural_bp(
            H_l, G_l,
            num_iter=5, snr_db_list=[3.0, 5.0, 7.0],
            num_epochs=25, lr=0.08, batch_size=12,
            epsilon=0.06, verbose=False,
        )
        print(f" 权重={np.round(w_l, 3)}")

        b_t = simulate_ber_minsum(H_l, G_l, snr_len, frames_len, max_iter=5)
        b_n = simulate_ber_neural(H_l, G_l, w_l, snr_len, frames_len)
        ber_trad_len.append(b_t)
        ber_neural_len.append(b_n)
        print(f"  传统MinSum BER={b_t:.5f}，Neural BP BER={b_n:.5f}")

    print("\n  绘制 BER vs 码长 图...")
    plot_ber_vs_length(lengths, ber_trad_len, ber_neural_len, snr_len, SAVE_DIR)

    # ────────────────────────────────────────────────
    # 第五步：BER vs 码率
    # ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("第五步：BER vs 码率（码长=32，SNR=5dB）")
    print("  为每种码率单独训练 Neural BP")
    print("=" * 60)

    n_rate = 32
    rate_ks = [8, 12, 16, 20, 24]    # 对应码率 0.25~0.75
    snr_rate = 5.0
    frames_rate = 150
    rates = [kk / n_rate for kk in rate_ks]
    ber_trad_rate, ber_neural_rate = [], []

    for kk in rate_ks:
        rate = kk / n_rate
        print(f"\n  [n={n_rate}, k={kk}, R={rate:.3f}] 训练 Neural BP...", end='', flush=True)
        G_r, H_r = make_ldpc_matrices(kk, n_rate - kk)
        w_r, _ = train_neural_bp(
            H_r, G_r,
            num_iter=5, snr_db_list=[3.0, 5.0, 7.0],
            num_epochs=25, lr=0.08, batch_size=12,
            epsilon=0.06, verbose=False,
        )
        print(f" 权重={np.round(w_r, 3)}")

        b_t = simulate_ber_minsum(H_r, G_r, snr_rate, frames_rate, max_iter=5)
        b_n = simulate_ber_neural(H_r, G_r, w_r, snr_rate, frames_rate)
        ber_trad_rate.append(b_t)
        ber_neural_rate.append(b_n)
        print(f"  传统MinSum BER={b_t:.5f}，Neural BP BER={b_n:.5f}")

    print("\n  绘制 BER vs 码率 图...")
    plot_ber_vs_rate(rates, ber_trad_rate, ber_neural_rate, n_rate, snr_rate, SAVE_DIR)

    # ────────────────────────────────────────────────
    # 结果汇总
    # ────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("仿真完毕！结果汇总")
    print("=" * 60)
    print(f"\n  总耗时：{elapsed:.1f} 秒\n")

    print("  【Neural BP 核心结论】")
    print(f"  训练后权重：{np.round(weights_main, 3)}")
    print("  权重 < 1.0：MinSum 高估了消息幅度，归一化后更准确")
    print("  权重在不同迭代间不同：不同轮次的消息可靠性不同")
    print()

    # 统计 BER 改善
    improvement = []
    for b_t, b_n in zip(ber_trad_snr, ber_neural_snr):
        if b_t > 1e-5 and b_n > 1e-5:
            improvement.append(b_t / b_n)
    if improvement:
        avg_imp = np.mean(improvement)
        print(f"  有效SNR范围内 BER 平均改善倍数：{avg_imp:.2f}×")

    print()
    print("  生成的图表文件：")
    for fname in ['neural_training.png', 'neural_ber_vs_snr.png',
                  'neural_snr_gain.png', 'neural_ber_vs_length.png',
                  'neural_ber_vs_rate.png']:
        print(f"    {os.path.join(SAVE_DIR, fname)}")

    print()
    print("  【算法对比说明】")
    print("  传统 MinSum：weights = [1, 1, 1, 1, 1]（无学习）")
    print("  Neural BP ：weights 由 Adam + 数值梯度训练得到")
    print("  本质区别   ：Neural BP 学会了对每轮消息的最优归一化")
    print("  理论依据   ：MinSum ≈ BP 的下界，归一化修正其过估计偏差")
    print("=" * 60)


if __name__ == "__main__":
    main()