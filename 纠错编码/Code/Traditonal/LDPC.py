"""
ldpc.py - LDPC码实现
包含：生成矩阵构造、编码、MinSum解码、验证
"""

import numpy as np


# ===== 矩阵构造 =====

def make_ldpc_matrices(k, m):
    """
    构造LDPC码的生成矩阵G和校验矩阵H。

    使用系统码结构：
        G = [I_k | P]        形状 (k, n), n = k + m
        H = [P^T | I_m]      形状 (m, n)
    满足 H · G^T = 0 (mod 2)，即任意合法码字均满足校验方程。

    参数：
        k : 信息位长度
        m : 校验位长度（即校验方程数量）
    返回：
        G : 生成矩阵 (k, n)
        H : 校验矩阵 (m, n)
    """
    n = k + m

    # 随机生成奇偶校验部分 P，形状 (k, m)
    P = np.random.randint(0, 2, size=(k, m))

    # 生成矩阵 G = [I_k | P]
    G = np.zeros((k, n), dtype=int)
    G[:, :k] = np.eye(k, dtype=int)
    G[:, k:] = P

    # 校验矩阵 H = [P^T | I_m]
    H = np.zeros((m, n), dtype=int)
    H[:, :k] = P.T
    H[:, k:] = np.eye(m, dtype=int)

    return G, H


# ===== 编码 =====

def ldpc_encode(message, G):
    """
    LDPC编码：c = message · G  (mod 2)

    参数：
        message : 信息位，形状 (k,)
        G       : 生成矩阵，形状 (k, n)
    返回：
        codeword : 码字，形状 (n,)

    复杂度：O(k · n)
    """
    codeword = np.dot(message, G) % 2
    return codeword.astype(int)


# ===== MinSum解码 =====

def ldpc_decode(llr, H, max_iter=10):
    """
    MinSum解码（BP算法的近似）。

    核心思想：
        - 变量节点（码字位）和校验节点（约束方程）之间迭代传递消息
        - 校验节点用最小值和符号积更新（代替BP中的tanh）
        - 变量节点累加所有校验节点的反馈
        - 每轮检查校验方程是否全满足，满足则提前退出

    参数：
        llr      : 信道对数似然比，形状 (n,)，正值倾向0，负值倾向1
        H        : 校验矩阵，形状 (m, n)
        max_iter : 最大迭代次数
    返回：
        decoded : 硬判决码字，形状 (n,)

    复杂度：O(max_iter · E)，E = H中1的总个数（边数）
    """
    m, n = H.shape

    # 预先计算每行/列的邻居索引，避免在迭代中重复搜索
    row_neighbors = [np.where(H[i, :] == 1)[0] for i in range(m)]
    col_neighbors = [np.where(H[:, j] == 1)[0] for j in range(n)]

    # 初始化消息矩阵
    # q[i,j]：变量节点j发往校验节点i的消息，初始为信道LLR
    # r[i,j]：校验节点i发往变量节点j的消息，初始为0
    q = np.zeros((m, n))
    r = np.zeros((m, n))
    for i in range(m):
        for j in row_neighbors[i]:
            q[i, j] = llr[j]

    for _ in range(max_iter):
        # --- 校验节点更新（MinSum规则）---
        # 对校验节点i的每个邻居变量节点j：
        #   符号 = 其余邻居消息的符号之积
        #   幅度 = 其余邻居消息绝对值的最小值
        for i in range(m):
            nbrs = row_neighbors[i]
            for j in nbrs:
                others = nbrs[nbrs != j]
                if len(others) == 0:
                    r[i, j] = 0
                    continue
                signs = np.sign(q[i, others])
                signs[signs == 0] = 1
                r[i, j] = np.prod(signs) * np.min(np.abs(q[i, others]))

        # --- 变量节点更新 ---
        # 变量节点j发往校验节点i的消息 = 信道LLR + 其余校验节点的反馈之和
        for j in range(n):
            nbrs = col_neighbors[j]
            for i in nbrs:
                others = nbrs[nbrs != i]
                q[i, j] = llr[j] + np.sum(r[others, j])

        # --- 硬判决：后验LLR = 信道LLR + 所有校验节点反馈之和 ---
        posterior = llr + np.sum(r, axis=0)
        decoded = (posterior < 0).astype(int)

        # --- 提前终止：若所有校验方程满足则解码成功 ---
        syndrome = np.dot(H, decoded) % 2
        if np.all(syndrome == 0):
            return decoded

    # 达到最大迭代次数，返回当前硬判决
    return (llr + np.sum(r, axis=0) < 0).astype(int)


# ===== 验证 =====

def validate_ldpc(k, m):
    """
    验证LDPC码的以下属性，并打印结果：
        1. 矩阵维度是否正确
        2. H · G^T = 0 (mod 2)（校验矩阵与生成矩阵正交）
        3. 编码码字满足校验方程 H · c = 0 (mod 2)
        4. 系统码前k位与原始消息一致
        5. 零噪声场景下解码结果与原始消息完全一致

    参数：
        k : 信息位长度
        m : 校验位长度
    """
    n = k + m
    G, H = make_ldpc_matrices(k, m)
    message = np.random.randint(0, 2, k)
    codeword = ldpc_encode(message, G)

    print("=" * 50)
    print("【LDPC码验证】")
    print(f"  码长n={n}，信息位k={k}，校验位m={m}，码率R={k/n:.2f}")
    print("-" * 50)

    # 验证1：维度
    ok = (G.shape == (k, n)) and (H.shape == (m, n))
    print(f"  ✓ 矩阵维度正确：G={G.shape}，H={H.shape}" if ok
          else f"  ✗ 矩阵维度错误")

    # 验证2：H · G^T = 0 (mod 2)
    product = np.dot(H, G.T) % 2
    ok = np.all(product == 0)
    print(f"  ✓ H·G^T = 0 (mod 2) 成立" if ok
          else f"  ✗ H·G^T ≠ 0，矩阵构造有误")

    # 验证3：码字满足校验方程
    syndrome = np.dot(H, codeword) % 2
    ok = np.all(syndrome == 0)
    print(f"  ✓ 码字满足所有{m}个校验方程" if ok
          else f"  ✗ 码字不满足校验方程，伴随式非零")

    # 验证4：系统码前k位与消息一致
    ok = np.all(codeword[:k] == message)
    print(f"  ✓ 系统位（前{k}位）与消息完全一致" if ok
          else f"  ✗ 系统位与消息不一致")

    # 验证5：零噪声下的解码
    # 零噪声时，LLR直接由码字决定（+∞ 或 -∞），用大数近似
    llr_noiseless = np.where(codeword == 0, 10.0, -10.0)
    decoded = ldpc_decode(llr_noiseless, H, max_iter=10)
    errors = int(np.sum(message != decoded[:k]))
    ok = (errors == 0)
    print(f"  ✓ 零噪声解码：误比特数 0/{k}，完全正确" if ok
          else f"  ✗ 零噪声解码：误比特数 {errors}/{k}")

    print("=" * 50)
    return ok