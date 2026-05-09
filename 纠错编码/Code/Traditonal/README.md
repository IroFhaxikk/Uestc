# 5G 信道编码：LDPC 码与 Polar 码

## 项目结构

```
5g_coding/
├── ldpc.py    # LDPC码：矩阵构造、编码、MinSum解码、验证
├── polar.py   # Polar码：可靠位选择、蝶形编码、SC解码、验证
├── main.py    # 主程序：验证 → 复杂度分析 → BER仿真 → 画图
└── README.md  # 本文档
```

运行方式：

```bash
cd 5g_coding
python main.py
```

---

## `ldpc.py` 函数说明

### `make_ldpc_matrices(k, m)`

**作用**：构造LDPC码的生成矩阵G和校验矩阵H。

使用系统码结构，随机生成奇偶校验部分P：
- `G = [I_k | P]`，形状 `(k, n)`，前k列是单位阵（系统位），后m列是P（校验位）
- `H = [P^T | I_m]`，形状 `(m, n)`
- 满足 `H · G^T = 0 (mod 2)`，即任意合法码字均满足所有校验方程

| 参数 | 说明 |
|------|------|
| `k` | 信息位长度 |
| `m` | 校验位长度（m = n - k） |
| 返回 | `(G, H)` 两个矩阵 |

---

### `ldpc_encode(message, G)`

**作用**：LDPC编码，计算 `c = message · G (mod 2)`。

- 输入信息位向量与生成矩阵做模2矩阵乘法
- 输出包含系统位和校验位的完整码字
- **复杂度**：O(k · n)

| 参数 | 说明 |
|------|------|
| `message` | 信息位，形状 `(k,)` |
| `G` | 生成矩阵，形状 `(k, n)` |
| 返回 | 码字，形状 `(n,)` |

---

### `ldpc_decode(llr, H, max_iter=10)`

**作用**：MinSum解码（置信传播BP算法的近似）。

**算法流程**：
1. 用信道LLR初始化变量节点消息
2. **校验节点更新**：对每个校验节点，用其邻居变量节点消息的符号积和最小绝对值计算反馈
3. **变量节点更新**：信道LLR加上所有校验节点反馈之和
4. 硬判决，检查校验方程是否全满足 → 满足则提前终止
5. 未收敛则继续迭代，直到 `max_iter` 次

**复杂度**：O(max_iter · E)，E = H中1的总个数（稀疏矩阵保证E远小于m·n）

| 参数 | 说明 |
|------|------|
| `llr` | 信道对数似然比，形状 `(n,)`，正值倾向0 |
| `H` | 校验矩阵，形状 `(m, n)` |
| `max_iter` | 最大迭代次数，默认10 |
| 返回 | 硬判决码字，形状 `(n,)` |

---

### `validate_ldpc(k, m)`

**作用**：验证LDPC码实现的正确性，打印逐项检验结果。

验证项：
1. 矩阵维度是否正确
2. `H · G^T = 0 (mod 2)`（数学正确性）
3. 编码码字满足校验方程 `H · c = 0 (mod 2)`
4. 系统码前k位与原始消息一致
5. 零噪声场景下解码误比特数为0

---

## `polar.py` 函数说明

### `get_frozen_and_info(N, K)`

**作用**：选择K个最可靠的位置作为信息位，其余N-K个为冻结位（固定为0）。

**可靠性判据**：汉明重量越大（二进制表示中1越多）的位置越可靠，优先选为信息位。这是信道极化理论的近似实现。

| 参数 | 说明 |
|------|------|
| `N` | 码长，必须是2的幂 |
| `K` | 信息位数量 |
| 返回 | `(info_pos, frozen_set)`：信息位索引列表和冻结位集合 |

---

### `polar_encode(message, N, info_pos)`

**作用**：Polar码编码，执行蝶形变换 `x = u · F^⊗n (mod 2)`。

**算法步骤**：
1. 将K个信息位放入u向量的指定位置，冻结位填0
2. 执行 `n = log₂N` 层蝶形运算，每层对相距 `stride=2^l` 的位对做异或

**复杂度**：O(N · log N)

| 参数 | 说明 |
|------|------|
| `message` | 信息位，形状 `(K,)` |
| `N` | 码长 |
| `info_pos` | 信息位在u向量中的索引列表 |
| 返回 | 码字，形状 `(N,)` |

---

### `_f_op(la, lb)` （内部函数）

**作用**：SC解码中第一子信道的LLR合并（MinSum近似）。

精确公式：`log[(1 + e^(La+Lb)) / (e^La + e^Lb)]`  
近似公式：`sign(La) · sign(Lb) · min(|La|, |Lb|)`

对应码字第一半的"XOR约束"：若 x = a⊕b 和 y = b，则 La 综合了 a 的可靠信息。

---

### `_g_op(la, lb, u_hat)` （内部函数）

**作用**：SC解码中第二子信道的LLR计算（已知第一子信道判决）。

公式：`g(La, Lb, u) = (1 - 2u) · La + Lb`

- `u=0` 时：`g = La + Lb`（两个信道直接加强）
- `u=1` 时：`g = Lb - La`（第一个信道"翻转"）

---

### `_sc_recursive(llr, frozen_set, offset)` （内部函数）

**作用**：SC解码的递归核心，将长度N的问题分解为两个N/2的子问题。

**递归结构**：
```
sc_recursive(llr, N):
    ├── left:  f_llr = f(llr[:N/2], llr[N/2:])
    │           u_left = sc_recursive(f_llr, N/2)
    └── right: g_llr = g(llr[:N/2], llr[N/2:], u_left)
                u_right = sc_recursive(g_llr, N/2)
```
基础情况（N=1）：直接硬判决（冻结位强制为0）。

**复杂度**：T(N) = 2·T(N/2) + O(N)，解为 O(N·log N)

---

### `polar_decode(llr, N, frozen_set, info_pos)`

**作用**：Polar码SC（连续消除）解码的对外接口。

调用 `_sc_recursive` 得到完整的u_hat向量，然后提取信息位位置的值作为解码结果。

| 参数 | 说明 |
|------|------|
| `llr` | 信道LLR，形状 `(N,)` |
| `N` | 码长 |
| `frozen_set` | 冻结位索引集合 |
| `info_pos` | 信息位索引列表 |
| 返回 | 解码信息位，形状 `(K,)` |

---

### `validate_polar(N, K)`

**作用**：验证Polar码实现的正确性，打印逐项检验结果。

验证项：
1. N是2的幂
2. 冻结位数量 = N - K
3. 码字长度 = N
4. 冻结位在u向量中值为0
5. 零噪声场景下解码误比特数为0
6. **环路验证**：对解码结果重新编码，应与原始码字完全一致

---

## `main.py` 函数说明

### `awgn_llr(codeword, snr_db)`

**作用**：AWGN信道仿真，返回信道LLR。

- BPSK调制：0→+1，1→-1
- 加高斯噪声：n ~ N(0, σ²)，σ² = 1/SNR
- LLR公式：L = 2y/σ²（正值倾向0，负值倾向1）

---

### `simulate_ber_ldpc(n, k, snr_db, num_frames=50)`

**作用**：在指定参数下仿真LDPC码的误比特率。

对每帧：随机生成消息 → 编码 → 过AWGN信道 → 解码 → 统计误比特。

---

### `simulate_ber_polar(N, K, snr_db, num_frames=50)`

**作用**：在指定参数下仿真Polar码的误比特率，流程同上。

---

### `print_complexity()`

**作用**：打印LDPC和Polar码编解码算法的复杂度对比分析。

---

### `plot_ber_vs_snr(...)`

**作用**：绘制BER-SNR曲线（固定码长和码率，变化SNR），保存为 `ber_vs_snr.png`。

---

### `plot_ber_vs_length(...)`

**作用**：绘制BER随码长变化曲线（固定码率和SNR），保存为 `ber_vs_length.png`。

---

### `plot_ber_vs_rate(...)`

**作用**：绘制BER随码率变化曲线（固定码长和SNR），保存为 `ber_vs_rate.png`。

---

### `plot_complexity_comparison(save_dir)`

**作用**：绘制不同码长下LDPC和Polar的理论计算量对比，保存为 `complexity.png`。

---

### `main()`

**作用**：主流程，按顺序执行：

| 步骤 | 内容 |
|------|------|
| 第一步 | 验证LDPC和Polar码的编解码正确性 |
| 第二步 | 打印复杂度分析 + 生成复杂度对比图 |
| 第三步 | 仿真BER vs SNR（n=32，R=0.5） |
| 第四步 | 仿真BER vs 码长（R=0.5，SNR=5dB） |
| 第五步 | 仿真BER vs 码率（n=32，SNR=5dB） |

---

## 算法原理简述

### LDPC码 MinSum解码

MinSum是BP（置信传播）算法的简化：用 `min(|La|, |Lb|)` 代替精确的 `arctanh(tanh(La/2)·tanh(Lb/2))`，避免昂贵的非线性运算，性能损失约0.2~0.5 dB。

### Polar码 SC解码

SC（连续消除）解码基于极化信道的递归结构：
- f操作对应"XOR信道"的合并，得到第一子信道的LLR
- g操作利用已知判决"消除干扰"，得到第二子信道的LLR
- 从最不可靠（冻结）位到最可靠位依次判决，不可回溯

### 信道LLR

BPSK-AWGN信道下：`L = 2y/σ²`

- L > 0：倾向判为0（因为发+1时接收值倾向正数）
- L < 0：倾向判为1
- |L| 越大，置信度越高

---

## 注意事项

- 图表标签使用英文（避免WSL/Linux环境下matplotlib中文字体问题）
- 终端输出使用中文
- LDPC矩阵为教学用随机构造，非5G标准的准循环（QC-LDPC）
- Polar码SC解码为简化实现，实际5G使用SCL+CRC
- 默认每个仿真点50帧，可在 `main.py` 中调整 `frames` 参数