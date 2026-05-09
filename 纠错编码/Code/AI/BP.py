"""
neural_bp.py — Neural BP（神经置信传播）解码器
AI 模型：加权归一化 MinSum（Weighted Normalized MinSum）
编码方案：LDPC 码

核心思想：
    传统 MinSum 的每条消息权重固定为 1。
    MinSum 是 BP（tanh 规则）的近似，系统性地高估消息幅度，
    引入约 0.5–1.5 dB 的性能损失。

    Neural BP 为每次迭代引入可学习标量权重 w_t（t=0..T-1），
    通过梯度下降自动学习最优归一化因子，
    相当于让网络"知道"每一轮消息应该被放大还是缩小。

    当 weights = [1, 1, ..., 1] 时，退化为标准 MinSum。
    训练收敛后，weights 通常 < 1（0.7~0.9），
    体现了对 MinSum 过估计偏差的自动修正。

参考文献：
    Nachmani et al., "Learning to Decode Linear Codes", NeurIPS 2016
    Samuel et al., "Learning to Detect", IEEE TCOM 2019
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from LDPC import make_ldpc_matrices, ldpc_encode


# ===== 邻居预计算 =====

def precompute_neighbors(H):
    """
    预计算 H 的行邻居（每个校验节点的变量邻居）
    和列邻居（每个变量节点的校验邻居）。

    在训练和解码的每次迭代中反复调用，预计算避免重复 where 搜索。
    """
    m, n = H.shape
    row_nbrs = [np.where(H[i, :] == 1)[0] for i in range(m)]
    col_nbrs = [np.where(H[:, j] == 1)[0] for j in range(n)]
    return row_nbrs, col_nbrs


# ===== 核心解码函数 =====

def _check_node_update(q_row, weight):
    """
    对单个校验节点（行）做加权 MinSum 更新。

    数学公式：
        r_{i→j} = w · sign(∏_{k≠j} q_{k→i}) · min_{k≠j} |q_{k→i}|

    优化：
        sign 乘积 = 全体符号积 × 当前符号（除去自身等价于再乘一次）
        min 的计算：仅需保存最小值和次小值，O(d) 时间（d 为节点度数）
    """
    d = len(q_row)
    if d == 0:
        return np.zeros(0)

    abs_q = np.abs(q_row)
    signs = np.sign(q_row)
    signs[signs == 0] = 1
    total_sign = np.prod(signs)  # 所有符号的乘积

    # 找最小值和次小值（用于 O(d) 的 min-excluding-self）
    idx_min = int(np.argmin(abs_q))
    min1 = abs_q[idx_min]
    if d > 1:
        mask = np.ones(d, dtype=bool)
        mask[idx_min] = False
        min2 = float(np.min(abs_q[mask]))
    else:
        min2 = min1

    # 若最小值不唯一（出现两次以上），排除任一个后最小值仍为 min1
    min1_count = int(np.sum(abs_q == min1))

    r_row = np.empty(d)
    for k in range(d):
        sign_others = total_sign * signs[k]  # 除去 k 后的符号积
        if min1_count > 1 or abs_q[k] != min1:
            min_others = min1
        else:
            min_others = min2
        r_row[k] = weight * sign_others * min_others

    return r_row


def fast_weighted_minsum(llr, H, row_nbrs, col_nbrs, weights, num_iter):
    """
    加权 MinSum 解码（Neural BP 的前向计算）。

    与传统 MinSum 对比：
        校验节点更新后乘以可学习权重 weights[t]
        当 weights=[1,...,1] 时完全等价于传统 MinSum

    变量节点更新优化：
        q[i,j] = llr[j] + sum_all_r[:,j] - r[i,j]
        避免对"除 i 以外所有校验节点"的内层循环

    参数：
        llr      : 信道 LLR，形状 (n,)，正值→倾向0，负值→倾向1
        H        : 校验矩阵，形状 (m, n)
        row_nbrs : 每行的变量邻居索引列表（预计算）
        col_nbrs : 每列的校验邻居索引列表（预计算）
        weights  : 每次迭代的标量权重，形状 (num_iter,)
        num_iter : 迭代次数
    返回：
        decoded  : 硬判决码字，形状 (n,)
        posterior: 后验 LLR，形状 (n,)
    """
    m, n = H.shape
    # 初始化：变量节点→校验节点的消息 q[i,j] = 信道 LLR
    q = np.tile(llr, (m, 1))           # (m, n)
    r = np.zeros((m, n))               # 校验节点→变量节点的消息

    for t in range(num_iter):
        # ── 校验节点更新 ──
        for i in range(m):
            nbrs = row_nbrs[i]
            if len(nbrs) < 2:
                continue
            r_vals = _check_node_update(q[i, nbrs], weights[t])
            r[i, nbrs] = r_vals

        # ── 变量节点更新（向量化列求和，O(m·n) → O(n)）──
        sum_r = r.sum(axis=0)           # 每个变量节点收到的所有校验反馈之和
        for j in range(n):
            for i in col_nbrs[j]:
                # 排除自身：q[i,j] = llr[j] + 其他校验节点反馈之和
                q[i, j] = llr[j] + sum_r[j] - r[i, j]

    posterior = llr + r.sum(axis=0)
    decoded = (posterior < 0).astype(int)
    return decoded, posterior


# ===== 损失函数 =====

def bce_loss(posterior, true_bits):
    """
    二元交叉熵损失（Binary Cross-Entropy, BCE）。

    用途：
        BER（阶跃函数）不可微，无法直接用于梯度计算。
        BCE 是 BER 的平滑代理损失，梯度稳定。

    公式：
        L = -mean[ y · log σ(-L̃) + (1-y) · log σ(L̃) ]
    其中 L̃ 为后验 LLR，σ 为 sigmoid 函数，y 为真实比特（0 或 1）。

    物理含义：
        后验 LLR > 0 时认为 bit=0，符合真实标签则损失小。
    """
    # σ(-L̃) = P(bit=1)，σ(L̃) = P(bit=0)
    prob1 = 1.0 / (1.0 + np.exp(posterior))       # P(bit=1)
    prob1 = np.clip(prob1, 1e-9, 1 - 1e-9)
    prob0 = 1.0 - prob1
    loss = -np.mean(true_bits * np.log(prob1) + (1 - true_bits) * np.log(prob0))
    return float(loss)


# ===== 批量损失评估 =====

def compute_batch_loss(weights, H, G, row_nbrs, col_nbrs,
                       snr_db, batch_size, num_iter):
    """
    在一个 mini-batch 上计算平均 BCE 损失。

    每次调用生成 batch_size 条随机消息 → 编码 → 加噪 → 加权 MinSum 解码 → 计算 BCE。
    训练时用作梯度数值估计的目标函数。
    """
    m, n = H.shape
    k = n - m
    snr_linear = 10 ** (snr_db / 10.0)
    sigma2 = 1.0 / snr_linear

    total_loss = 0.0
    for _ in range(batch_size):
        msg = np.random.randint(0, 2, k)
        codeword = ldpc_encode(msg, G)
        # BPSK 调制 + AWGN
        x = 1.0 - 2.0 * codeword
        noise = np.random.normal(0, np.sqrt(sigma2), n)
        llr = 2.0 * (x + noise) / sigma2
        _, posterior = fast_weighted_minsum(
            llr, H, row_nbrs, col_nbrs, weights, num_iter
        )
        total_loss += bce_loss(posterior, codeword.astype(float))

    return total_loss / batch_size


# ===== 训练主函数 =====

def train_neural_bp(H, G, num_iter=5, snr_db_list=None,
                    num_epochs=40, lr=0.08, batch_size=15,
                    epsilon=0.06, verbose=True):
    """
    训练 Neural BP 的每轮权重。

    优化算法：Adam（数值梯度版本）
    梯度估计：中心差分法（central finite differences）
        ∂L/∂w_t ≈ [L(w_t+ε) - L(w_t-ε)] / (2ε)

    多 SNR 训练策略：
        在 snr_db_list 中随机采样 SNR 进行训练，
        使解码器在不同信道质量下均有效（抗 SNR 过拟合）。

    参数：
        H, G         : LDPC 校验矩阵和生成矩阵
        num_iter     : BP 迭代次数（即权重个数）
        snr_db_list  : 训练用 SNR 列表（dB），默认 [2, 4, 6]
        num_epochs   : 训练轮数
        lr           : Adam 初始学习率
        batch_size   : 每次损失评估的样本数
        epsilon      : 数值梯度扰动步长
        verbose      : 是否打印训练进度
    返回：
        weights    : 训练后的权重，形状 (num_iter,)
        loss_curve : 每轮的 BCE 损失列表
    """
    if snr_db_list is None:
        snr_db_list = [2.0, 4.0, 6.0]

    row_nbrs, col_nbrs = precompute_neighbors(H)

    # 初始值 = 1.0（等价于传统 MinSum）
    weights = np.ones(num_iter, dtype=float)

    # Adam 优化器状态变量
    m1 = np.zeros(num_iter)     # 一阶矩估计
    v1 = np.zeros(num_iter)     # 二阶矩估计
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8

    loss_curve = []

    for epoch in range(num_epochs):
        snr_train = float(np.random.choice(snr_db_list))

        # ── 数值梯度（中心差分）──
        grad = np.zeros(num_iter)
        for i in range(num_iter):
            w_p = weights.copy(); w_p[i] += epsilon
            w_m = weights.copy(); w_m[i] -= epsilon
            lp = compute_batch_loss(w_p, H, G, row_nbrs, col_nbrs,
                                    snr_train, batch_size, num_iter)
            lm = compute_batch_loss(w_m, H, G, row_nbrs, col_nbrs,
                                    snr_train, batch_size, num_iter)
            grad[i] = (lp - lm) / (2.0 * epsilon)

        # ── Adam 更新 ──
        step = epoch + 1
        m1 = beta1 * m1 + (1 - beta1) * grad
        v1 = beta2 * v1 + (1 - beta2) * grad ** 2
        m_hat = m1 / (1 - beta1 ** step)
        v_hat = v1 / (1 - beta2 ** step)
        weights -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)

        # 约束权重到合理范围（0.05 ~ 3.0）
        weights = np.clip(weights, 0.05, 3.0)

        # 记录当前损失（固定 SNR=4 dB 用于对比）
        cur_loss = compute_batch_loss(
            weights, H, G, row_nbrs, col_nbrs, 4.0, batch_size, num_iter
        )
        loss_curve.append(cur_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    第{epoch+1:3d}轮 | 损失={cur_loss:.5f} | "
                  f"权重={np.round(weights, 3)}")

    return weights, loss_curve


# ===== BER 仿真 =====

def simulate_ber_neural(H, G, weights, snr_db, num_frames=200):
    """
    仿真 Neural BP 解码器在指定 SNR 下的误比特率。

    仿真流程：随机消息 → LDPC 编码 → AWGN 信道 → 加权 MinSum 解码 → 统计错误

    参数：
        H, G       : LDPC 矩阵
        weights    : 训练好的 Neural BP 权重，形状 (num_iter,)
        snr_db     : 信噪比（dB）
        num_frames : 仿真帧数
    返回：
        ber : 误比特率（float）
    """
    m, n = H.shape
    k = n - m
    num_iter = len(weights)
    row_nbrs, col_nbrs = precompute_neighbors(H)
    snr_linear = 10 ** (snr_db / 10.0)
    sigma2 = 1.0 / snr_linear

    errors = 0
    total = 0
    for _ in range(num_frames):
        msg = np.random.randint(0, 2, k)
        codeword = ldpc_encode(msg, G)
        x = 1.0 - 2.0 * codeword
        noise = np.random.normal(0, np.sqrt(sigma2), n)
        llr = 2.0 * (x + noise) / sigma2
        decoded, _ = fast_weighted_minsum(
            llr, H, row_nbrs, col_nbrs, weights, num_iter
        )
        errors += int(np.sum(msg != decoded[:k]))
        total += k

    return errors / total if total > 0 else 0.0


# ===== 验证 =====

def validate_neural_bp(H, G, weights, num_iter=None):
    """
    验证 Neural BP 解码器的正确性：
        1. 权重数量与迭代次数一致
        2. 权重已偏离初始值 1.0（说明训练有效）
        3. 零噪声场景下解码误比特数为 0
    """
    m, n = H.shape
    k = n - m
    if num_iter is None:
        num_iter = len(weights)
    row_nbrs, col_nbrs = precompute_neighbors(H)

    print("=" * 50)
    print("【Neural BP 验证】")
    print(f"  码长n={n}，信息位k={k}，迭代次数={num_iter}")
    print("-" * 50)

    # 验证1：权重数量
    ok = len(weights) == num_iter
    print(f"  ✓ 权重数量正确：{num_iter} 个" if ok
          else f"  ✗ 权重数量错误：期望 {num_iter}，实际 {len(weights)}")

    # 验证2：权重偏离初始值（训练有效性）
    deviation = float(np.max(np.abs(weights - 1.0)))
    trained = deviation > 0.03
    print(f"  ✓ 权重已收敛，最大偏差={deviation:.4f}（训练有效）" if trained
          else f"  ⚠ 权重几乎未变（偏差={deviation:.4f}），可能训练不足")
    print(f"    训练后权重：{np.round(weights, 4)}")
    print(f"    初始权重  ：{np.ones(num_iter)}")

    # 验证3：零噪声解码
    msg = np.random.randint(0, 2, k)
    codeword = ldpc_encode(msg, G)
    llr_noiseless = np.where(codeword == 0, 20.0, -20.0)
    decoded, _ = fast_weighted_minsum(
        llr_noiseless, H, row_nbrs, col_nbrs, weights, num_iter
    )
    errs = int(np.sum(msg != decoded[:k]))
    ok = errs == 0
    print(f"  ✓ 零噪声解码：误比特数 0/{k}，完全正确" if ok
          else f"  ✗ 零噪声解码：误比特数 {errs}/{k}")

    print("=" * 50)
    return trained and ok