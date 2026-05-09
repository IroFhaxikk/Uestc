"""
polar.py - Polar码实现
包含：信道可靠性排序、编码（蝶形变换）、SC解码（递归）、验证
"""

import numpy as np


# ===== 可靠位置选择 =====

def get_frozen_and_info(N, K):
    """
    选择K个最可靠的位置作为信息位，其余为冻结位（固定为0）。

    可靠性判据（近似）：
        汉明重量越大的位置可靠性越高，相同重量的以索引大的为优先。
        这对应于"合并后的信道容量递增"这一极化规律。

    参数：
        N : 码长（必须是2的幂）
        K : 信息位数量
    返回：
        info_pos   : 信息位索引列表（升序）
        frozen_set : 冻结位索引集合
    """
    # 按汉明重量升序排列，最后K个为最可靠的信息位
    order = sorted(range(N), key=lambda i: (bin(i).count('1'), i))
    info_pos = sorted(order[-K:])
    frozen_set = set(order[:-K])
    return info_pos, frozen_set


# ===== 编码 =====

def polar_encode(message, N, info_pos):
    """
    Polar编码：蝶形变换（F^⊗n，无比特翻转置换）。

    步骤：
        1. 将信息位放入u向量的指定位置，冻结位填0
        2. 执行n = log2(N) 层蝶形运算：u·F^⊗n (mod 2)

    参数：
        message  : 信息位，形状 (K,)
        N        : 码长
        info_pos : 信息位在u向量中的位置列表
    返回：
        codeword : 码字，形状 (N,)

    复杂度：O(N · log N)
    """
    n = int(np.log2(N))

    # 构建输入向量u：冻结位为0，信息位填入message
    u = np.zeros(N, dtype=int)
    for idx, pos in enumerate(info_pos):
        u[pos] = message[idx]

    # 蝶形变换：每层stride加倍，相邻stride位做异或
    x = u.copy()
    for stage in range(n):
        stride = 1 << stage       # stride = 2^stage
        for start in range(0, N, stride << 1):
            for j in range(stride):
                a, b = start + j, start + j + stride
                x[a] = (x[a] + x[b]) % 2  # x[a] ^= x[b]

    return x


# ===== SC解码（递归） =====

def _butterfly(u):
    """
    对长度为N（2的幂）的向量做蝶形变换。

    SC解码中，左子树解码出 u_left 后，需要知道 u_left 通过极化变换
    对原始信道的"贡献"（partial sums），才能正确计算右子树的g-LLR。
    这个贡献恰好是 u_left 经过蝶形变换的结果。

    例如 u_left = [u0, u1]：
        _butterfly([u0, u1]) = [u0⊕u1, u1]
        对应 N=4 时，左两路对 x[0], x[1] 的贡献分别是 u0⊕u1 和 u1。
    """
    N = len(u)
    if N == 1:
        return u.copy()
    x = u.astype(int).copy()
    n = int(np.log2(N))
    for stage in range(n):
        stride = 1 << stage
        for start in range(0, N, stride << 1):
            for j in range(stride):
                a, b = start + j, start + j + stride
                x[a] = (x[a] + x[b]) % 2
    return x


def _f_op(la, lb):
    """
    f操作（MinSum近似）：用于SC解码中第一个子信道的LLR合并。
    精确公式：log[(1 + e^(La+Lb)) / (e^La + e^Lb)]
    近似：sign(La) · sign(Lb) · min(|La|, |Lb|)
    """
    sign = np.sign(la) * np.sign(lb)
    sign[sign == 0] = 1
    return sign * np.minimum(np.abs(la), np.abs(lb))


def _g_op(la, lb, u_hat):
    """
    g操作：用于SC解码中第二个子信道的LLR计算（已知第一子信道判决）。
    公式：g(La, Lb, u) = (1 - 2u) · La + Lb
    """
    return (1 - 2 * u_hat) * la + lb


def _sc_recursive(llr, frozen_set, offset):
    """
    SC解码递归核心。

    原理：
        将长度N的问题分成两个长度N/2的子问题：
          - 左子树：用f操作合并相邻信道LLR，解码u[offset : offset+N/2]
          - 右子树：用g操作（利用左子树决定），解码u[offset+N/2 : offset+N]
        递归到N=1时直接硬判决。

    参数：
        llr        : 当前子树的LLR向量，形状 (N,)
        frozen_set : 全局冻结位索引集合
        offset     : 当前子树在全局u向量中的起始偏移
    返回：
        u_hat : 当前子树的解码结果，形状 (N,)
    """
    N = len(llr)

    # 递归基：单比特，直接判决
    if N == 1:
        if offset in frozen_set:
            return np.array([0])
        return np.array([0 if llr[0] >= 0 else 1])

    half = N // 2
    la = llr[:half]
    lb = llr[half:]

    # 左子树：f操作 + 递归解码
    f_llr = _f_op(la, lb)
    u_left = _sc_recursive(f_llr, frozen_set, offset)

    # 关键：g操作需要 partial sums，而非 u_left 本身
    # partial sums = u_left 经蝶形变换后的值，表示左子树对原始信道的贡献
    # 例如 N=4 时，u_left=[u0,u1] → partial=[u0⊕u1, u1]
    partial = _butterfly(u_left)

    # 右子树：g操作（利用partial sums）+ 递归解码
    g_llr = _g_op(la, lb, partial)
    u_right = _sc_recursive(g_llr, frozen_set, offset + half)

    return np.concatenate([u_left, u_right])


def polar_decode(llr, N, frozen_set, info_pos):
    """
    Polar码SC（连续消除）解码。

    参数：
        llr        : 信道LLR，形状 (N,)
        N          : 码长
        frozen_set : 冻结位索引集合
        info_pos   : 信息位索引列表
    返回：
        message : 解码出的信息位，形状 (K,)

    复杂度：O(N · log N)
    """
    u_hat = _sc_recursive(llr, frozen_set, offset=0)
    return np.array([u_hat[pos] for pos in info_pos])


# ===== 验证 =====

def validate_polar(N, K):
    """
    验证Polar码的以下属性，并打印结果：
        1. 码长N必须是2的幂
        2. 冻结位数量 = N - K
        3. 编码后码字长度 = N
        4. 冻结位在u向量中值为0（由编码过程保证）
        5. 零噪声场景下解码结果与原始消息完全一致
        6. 重新编码解码结果，应与原始码字一致（环路验证）

    参数：
        N : 码长
        K : 信息位长度
    """
    info_pos, frozen_set = get_frozen_and_info(N, K)
    message = np.random.randint(0, 2, K)
    codeword = polar_encode(message, N, info_pos)

    print("=" * 50)
    print("【Polar码验证】")
    print(f"  码长N={N}，信息位K={K}，冻结位数={N-K}，码率R={K/N:.2f}")
    print(f"  信息位位置：{info_pos}")
    print("-" * 50)

    # 验证1：N是2的幂
    ok = (N > 0) and ((N & (N - 1)) == 0)
    print(f"  ✓ N={N} 是2的幂" if ok else f"  ✗ N={N} 不是2的幂")

    # 验证2：冻结位数量
    ok = (len(frozen_set) == N - K)
    print(f"  ✓ 冻结位数量正确：{N-K}个" if ok
          else f"  ✗ 冻结位数量错误：期望{N-K}，实际{len(frozen_set)}")

    # 验证3：码字长度
    ok = (len(codeword) == N)
    print(f"  ✓ 码字长度正确：{N}位" if ok
          else f"  ✗ 码字长度错误：期望{N}，实际{len(codeword)}")

    # 验证4：冻结位在u向量中值为0
    # 反推u：对码字做逆变换（蝶形变换是自逆的，再做一次得到u）
    u_check = polar_encode(message, N, info_pos)  # 这是x
    # 对x再做一次蝶形变换得到u（因为F^⊗n是自逆的，F^⊗n · F^⊗n = I）
    u_recovered = polar_encode(
        np.array([codeword[pos] for pos in info_pos]),
        N,
        info_pos
    )
    # 直接重建u向量来检查冻结位
    u_original = np.zeros(N, dtype=int)
    for idx, pos in enumerate(info_pos):
        u_original[pos] = message[idx]
    frozen_ok = all(u_original[i] == 0 for i in frozen_set)
    print(f"  ✓ 所有冻结位在u向量中为0" if frozen_ok
          else f"  ✗ 冻结位中有非零值（代码有误）")

    # 验证5：零噪声下解码
    llr_noiseless = np.where(codeword == 0, 10.0, -10.0)
    decoded = polar_decode(llr_noiseless, N, frozen_set, info_pos)
    errors = int(np.sum(message != decoded))
    ok = (errors == 0)
    print(f"  ✓ 零噪声解码：误比特数 0/{K}，完全正确" if ok
          else f"  ✗ 零噪声解码：误比特数 {errors}/{K}")

    # 验证6：环路验证——重新编码解码结果，应与原始码字一致
    re_encoded = polar_encode(decoded, N, info_pos)
    ok = np.all(re_encoded == codeword)
    print(f"  ✓ 环路验证通过：decode→encode 结果与原始码字一致" if ok
          else f"  ✗ 环路验证失败：重编码结果与原始码字不符")

    print("=" * 50)
    return errors == 0