"""
IMM (Interacting Multiple Model) 卡尔曼滤波目标跟踪仿真
场景：平面机动目标，匀速+慢转弯+恢复匀速+快转弯+匀速
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
# ── 中文字体配置（优先 SimHei，其次 WenQuanYi，最后内置 sans-serif）──
import matplotlib.font_manager as _fm
_cjk_candidates = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
                   'Noto Sans SC', 'Microsoft YaHei', 'PingFang SC',
                   'Heiti SC', 'STSong', 'Source Han Sans CN']
_available = {f.name for f in _fm.fontManager.ttflist}
_cjk_font  = next((f for f in _cjk_candidates if f in _available), None)
if _cjk_font:
    matplotlib.rcParams['font.family']        = [_cjk_font, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    # 找不到 CJK 字体时退回英文标签，不报 Warning
    matplotlib.rcParams['font.family']        = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    _cjk_font = None
print(f"[字体] 使用: {_cjk_font or 'DejaVu Sans（无中文字体，标签将为英文）'}")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import pathlib, os
OUT_DIR = pathlib.Path(__file__).parent / 'Image'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[输出] 图形保存到: {OUT_DIR.resolve()}")

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. 仿真参数
# ─────────────────────────────────────────────
T       = 2.0        # 采样间隔 (s)
sigma_r = 100.0      # 量测噪声标准差 (m)
t_end   = 800.0      # 仿真总时长 (s)
times   = np.arange(0, t_end + T/2, T)
N       = len(times)

v0           = 12.0                  # 初始速度大小 (m/s)
omega_slow   =  0.075 / v0          # 慢转弯角速率  ≈ 0.00625 rad/s
omega_fast   = -0.3   / v0          # 快转弯角速率  ≈ -0.025  rad/s

print(f"仿真步数 N = {N}")
print(f"慢转弯角速率 = {omega_slow:.5f} rad/s")
print(f"快转弯角速率 = {omega_fast:.5f} rad/s")

# ─────────────────────────────────────────────
# 2. 协调转弯 (CT) 状态转移
# ─────────────────────────────────────────────
def ct_transition(state, omega, dt):
    """
    状态: [x, vx, y, vy]
    协调转弯方程（omega=0 退化为匀速）
    """
    x, vx, y, vy = state
    if abs(omega) < 1e-9:
        return np.array([x + vx*dt, vx, y + vy*dt, vy])
    swT = np.sin(omega * dt)
    cwT = np.cos(omega * dt)
    return np.array([
        x  +  swT/omega * vx - (1 - cwT)/omega * vy,
        cwT * vx - swT * vy,
        y  + (1 - cwT)/omega * vx + swT/omega * vy,
        swT * vx + cwT * vy
    ])

# ─────────────────────────────────────────────
# 3. 生成真实轨迹
# ─────────────────────────────────────────────
true_state = np.zeros((N, 4))
true_state[0] = [1000.0, 0.0, 8000.0, -12.0]

for k in range(N - 1):
    t = times[k]
    if   t <  400:  om = 0.0
    elif t <= 600:  om = omega_slow
    elif t <= 610:  om = 0.0
    elif t <= 660:  om = omega_fast
    else:           om = 0.0
    true_state[k+1] = ct_transition(true_state[k], om, T)

# ─────────────────────────────────────────────
# 4. 生成带噪声的量测
# ─────────────────────────────────────────────
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
R = (sigma_r**2) * np.eye(2)

noise     = np.random.multivariate_normal([0, 0], R, N)
measurements = true_state[:, [0, 2]] + noise   # (N, 2)

# ─────────────────────────────────────────────
# 5. IMM 滤波器设置
# ─────────────────────────────────────────────
n_models     = 3
model_omegas = [0.0, omega_slow, omega_fast]
model_names  = ['CV', 'CT(慢)', 'CT(快)']

def get_F_ct(omega, dt):
    """CT 状态转移矩阵"""
    if abs(omega) < 1e-9:
        return np.array([[1, dt, 0,  0],
                         [0,  1, 0,  0],
                         [0,  0, 1, dt],
                         [0,  0, 0,  1]], dtype=float)
    s   = np.sin(omega*dt) / omega
    c   = (1 - np.cos(omega*dt)) / omega
    cwT = np.cos(omega*dt)
    swT = np.sin(omega*dt)
    return np.array([[1,  s,  0, -c],
                     [0, cwT, 0, -swT],
                     [0,  c,  1,  s],
                     [0, swT, 0, cwT]], dtype=float)

def get_Q(sigma_a, dt):
    """Singer 过程噪声矩阵"""
    q = sigma_a**2
    return q * np.array([
        [dt**4/4, dt**3/2,       0,       0],
        [dt**3/2,    dt**2,      0,       0],
        [0,           0,  dt**4/4, dt**3/2],
        [0,           0,  dt**3/2,   dt**2]
    ])

Fs    = [get_F_ct(om, T) for om in model_omegas]
s_as  = [0.3, 0.8, 1.5]                       # 各模型过程噪声加速度标准差
Qs    = [get_Q(s, T) for s in s_as]

# Markov 转移矩阵（停留概率 0.8）
p_stay = 0.80
p_trans = (1 - p_stay) / (n_models - 1)
Pi = p_trans * np.ones((n_models, n_models))
np.fill_diagonal(Pi, p_stay)

# ─────────────────────────────────────────────
# 6. 初始化 IMM
# ─────────────────────────────────────────────
mu       = np.ones(n_models) / n_models          # 初始模型概率
x_m      = [true_state[0].copy() for _ in range(n_models)]
P_m      = [np.diag([200**2, 5**2, 200**2, 5**2]) for _ in range(n_models)]

imm_est  = np.zeros((N, 4))
imm_est[0] = true_state[0].copy()
all_mu   = np.zeros((N, n_models))
all_mu[0] = mu

# ─────────────────────────────────────────────
# 7. IMM 递推
# ─────────────────────────────────────────────
for k in range(1, N):
    z = measurements[k]

    # --- 7.1 交互（混合）---
    c_bar = Pi.T @ mu                             # 预测模型概率 (n_models,)
    mu_ij = np.zeros((n_models, n_models))        # mu_ij[i,j] = P(M_k-1=i | M_k=j, Z_k-1)
    for i in range(n_models):
        for j in range(n_models):
            if c_bar[j] > 1e-300:
                mu_ij[i, j] = Pi[i, j] * mu[i] / c_bar[j]

    x0 = []
    P0 = []
    for j in range(n_models):
        xj = np.zeros(4)
        for i in range(n_models):
            xj += mu_ij[i, j] * x_m[i]
        Pj = np.zeros((4, 4))
        for i in range(n_models):
            d = x_m[i] - xj
            Pj += mu_ij[i, j] * (P_m[i] + np.outer(d, d))
        x0.append(xj)
        P0.append(Pj)

    # --- 7.2 各模型卡尔曼滤波 ---
    x_upd = []
    P_upd = []
    Lk    = np.zeros(n_models)

    for j in range(n_models):
        # 预测
        xp = Fs[j] @ x0[j]
        Pp = Fs[j] @ P0[j] @ Fs[j].T + Qs[j]

        # 量测更新
        innov = z - H @ xp
        S     = H @ Pp @ H.T + R
        K     = Pp @ H.T @ np.linalg.inv(S)
        xu    = xp + K @ innov
        Pu    = (np.eye(4) - K @ H) @ Pp

        x_upd.append(xu)
        P_upd.append(Pu)

        # 似然
        sign, ldet = np.linalg.slogdet(2 * np.pi * S)
        Lk[j] = np.exp(-0.5 * (innov @ np.linalg.solve(S, innov)) - 0.5 * ldet)

    x_m, P_m = x_upd, P_upd

    # --- 7.3 更新模型概率 ---
    mu_new = c_bar * Lk
    s_mu   = mu_new.sum()
    mu     = mu_new / s_mu if s_mu > 1e-300 else c_bar
    all_mu[k] = mu

    # --- 7.4 总体融合估计 ---
    imm_est[k] = sum(mu[j] * x_m[j] for j in range(n_models))

# ─────────────────────────────────────────────
# 8. 误差统计
# ─────────────────────────────────────────────
err_x_imm  = imm_est[:, 0] - true_state[:, 0]
err_y_imm  = imm_est[:, 2] - true_state[:, 2]
err_x_meas = measurements[:, 0] - true_state[:, 0]
err_y_meas = measurements[:, 1] - true_state[:, 2]

rmse_x_imm  = np.sqrt(np.mean(err_x_imm**2))
rmse_y_imm  = np.sqrt(np.mean(err_y_imm**2))
rmse_x_meas = np.sqrt(np.mean(err_x_meas**2))
rmse_y_meas = np.sqrt(np.mean(err_y_meas**2))

print(f"\n{'─'*45}")
print(f"{'':10s} {'RMSE_X (m)':>12s}  {'RMSE_Y (m)':>12s}")
print(f"{'量测噪声':10s} {rmse_x_meas:>12.2f}  {rmse_y_meas:>12.2f}")
print(f"{'IMM滤波':10s} {rmse_x_imm:>12.2f}  {rmse_y_imm:>12.2f}")
print(f"{'─'*45}")

# ─────────────────────────────────────────────
# 9. 绘图
# ─────────────────────────────────────────────
COLORS = {'truth': '#1f4e79', 'meas': '#bfbfbf', 'imm': '#c00000'}
MSIZE  = 2

# ── Fig 1: 二维轨迹 ──────────────────────────
fig1, ax = plt.subplots(figsize=(9, 7))
ax.scatter(measurements[:, 0], measurements[:, 1],
           s=MSIZE, c=COLORS['meas'], alpha=0.5, label='量测值', zorder=1)
ax.plot(true_state[:, 0], true_state[:, 2],
        color=COLORS['truth'], lw=1.5, label='真实轨迹', zorder=3)
ax.plot(imm_est[:, 0], imm_est[:, 2],
        color=COLORS['imm'], lw=1.5, ls='--', label='IMM估计', zorder=4)

# 标注各段起止
seg_times = [0, 400, 600, 610, 660, t_end]
seg_labels = ['匀速起始', '慢转弯开始\n(a=0.075m/s²)',
              '恢复匀速', '快转弯开始\n(a=-0.3m/s²)', '末端匀速']
for st, lbl in zip(seg_times[1:-1], seg_labels[1:]):
    idx = int(st / T)
    ax.scatter(*true_state[idx, [0, 2]], s=60, color='orange',
               zorder=5, marker='D')
    ax.annotate(lbl, xy=true_state[idx, [0, 2]],
                xytext=(15, 10), textcoords='offset points',
                fontsize=7.5, color='darkorange',
                arrowprops=dict(arrowstyle='->', color='orange', lw=0.8))

ax.scatter(*true_state[0, [0, 2]], s=80, color='green', zorder=6, marker='s', label='初始位置')
ax.scatter(*true_state[-1, [0, 2]], s=80, color='red',  zorder=6, marker='*', label='终止位置')

ax.set_xlabel('X 位置 (m)', fontsize=12)
ax.set_ylabel('Y 位置 (m)', fontsize=12)
ax.set_title('平面机动目标 IMM 跟踪轨迹', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(str(OUT_DIR / 'fig1_trajectory.pdf'), dpi=150, bbox_inches='tight')
fig1.savefig(str(OUT_DIR / 'fig1_trajectory.png'), dpi=150, bbox_inches='tight')
print("Fig 1 saved.")

# ── Fig 2: 位置误差 vs 时间 ──────────────────
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# 标注机动段
maneuver_spans = [(400, 600, '#fff2cc', '慢转弯'),
                  (601, 610, '#e2efda', '匀速'),
                  (611, 660, '#fce4d6', '快转弯')]
for axj in (ax1, ax2):
    for ts, te, fc, lbl in maneuver_spans:
        axj.axvspan(ts, te, facecolor=fc, alpha=0.8, zorder=0)

ax1.plot(times, err_x_meas, color=COLORS['meas'], lw=0.8, alpha=0.7, label='量测误差')
ax1.plot(times, err_x_imm, color=COLORS['imm'], lw=1.2, label='IMM误差')
ax1.axhline(0, color='k', lw=0.5, ls='--')
ax1.set_ylabel('X 方向位置误差 (m)', fontsize=11)
ax1.legend(fontsize=10)
ax1.set_ylim(-350, 350)
ax1.grid(True, alpha=0.3)

ax2.plot(times, err_y_meas, color=COLORS['meas'], lw=0.8, alpha=0.7, label='量测误差')
ax2.plot(times, err_y_imm, color=COLORS['imm'], lw=1.2, label='IMM误差')
ax2.axhline(0, color='k', lw=0.5, ls='--')
ax2.set_ylabel('Y 方向位置误差 (m)', fontsize=11)
ax2.set_xlabel('时间 (s)', fontsize=11)
ax2.legend(fontsize=10)
ax2.set_ylim(-350, 350)
ax2.grid(True, alpha=0.3)

# 自定义图例说明机动段
custom_patches = [
    mpatches.Patch(facecolor='#fff2cc', alpha=0.8, label='慢转弯段 (400-600s)'),
    mpatches.Patch(facecolor='#fce4d6', alpha=0.8, label='快转弯段 (611-660s)'),
]
ax1.legend(handles=ax1.get_legend_handles_labels()[0] + custom_patches,
           fontsize=9, loc='upper left')

fig2.suptitle('位置误差对比（IMM 滤波 vs 原始量测）', fontsize=13, fontweight='bold')
fig2.tight_layout()
fig2.savefig(str(OUT_DIR / 'fig2_position_error.pdf'), dpi=150, bbox_inches='tight')
fig2.savefig(str(OUT_DIR / 'fig2_position_error.png'), dpi=150, bbox_inches='tight')
print("Fig 2 saved.")

# ── Fig 3: 模型概率 ───────────────────────────
fig3, ax = plt.subplots(figsize=(10, 4.5))
colors_m = ['#1f77b4', '#ff7f0e', '#2ca02c']
for j, (name, col) in enumerate(zip(model_names, colors_m)):
    ax.plot(times, all_mu[:, j], color=col, lw=1.5, label=name)

for ts, te, fc, lbl in maneuver_spans:
    ax.axvspan(ts, te, facecolor=fc, alpha=0.6, zorder=0, label=f'_{lbl}')

ax.set_xlabel('时间 (s)', fontsize=11)
ax.set_ylabel('模型概率', fontsize=11)
ax.set_title('IMM 模型概率演化', fontsize=13, fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, t_end)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 添加阶段标注
for ts, lbl in zip([0, 400, 601, 611, 660],
                   ['匀速', '慢转弯\nω=0.00625', '匀速', '快转弯\nω=-0.025', '匀速']):
    ax.axvline(ts, color='gray', lw=0.8, ls=':')

fig3.tight_layout()
fig3.savefig(str(OUT_DIR / 'fig3_model_probs.pdf'), dpi=150, bbox_inches='tight')
fig3.savefig(str(OUT_DIR / 'fig3_model_probs.png'), dpi=150, bbox_inches='tight')
print("Fig 3 saved.")

# ── Fig 4: 速度估计 ───────────────────────────
fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for axj in (ax1, ax2):
    for ts, te, fc, lbl in maneuver_spans:
        axj.axvspan(ts, te, facecolor=fc, alpha=0.8, zorder=0)

ax1.plot(times, true_state[:, 1], color=COLORS['truth'], lw=1.5, label='真实 Vx')
ax1.plot(times, imm_est[:, 1],   color=COLORS['imm'],   lw=1.2, ls='--', label='IMM估计 Vx')
ax1.set_ylabel('X 方向速度 (m/s)', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(times, true_state[:, 3], color=COLORS['truth'], lw=1.5, label='真实 Vy')
ax2.plot(times, imm_est[:, 3],   color=COLORS['imm'],   lw=1.2, ls='--', label='IMM估计 Vy')
ax2.set_ylabel('Y 方向速度 (m/s)', fontsize=11)
ax2.set_xlabel('时间 (s)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig4.suptitle('速度分量估计 vs 真实值', fontsize=13, fontweight='bold')
fig4.tight_layout()
fig4.savefig(str(OUT_DIR / 'fig4_velocity.pdf'), dpi=150, bbox_inches='tight')
fig4.savefig(str(OUT_DIR / 'fig4_velocity.png'), dpi=150, bbox_inches='tight')
print("Fig 4 saved.")

# ── Fig 5: 位置 RMSE 随时间滑动窗口 ─────────
win = 50
def sliding_rmse(err, w):
    out = np.full(len(err), np.nan)
    for i in range(w-1, len(err)):
        out[i] = np.sqrt(np.mean(err[i-w+1:i+1]**2))
    return out

fig5, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(times, sliding_rmse(err_x_meas, win), color=COLORS['meas'], lw=1.2,
        alpha=0.7, label=f'量测 RMSE_X (窗口={win}步)')
ax.plot(times, sliding_rmse(err_x_imm,  win), color=COLORS['imm'],  lw=1.5,
        label=f'IMM RMSE_X (窗口={win}步)')
ax.plot(times, sliding_rmse(err_y_meas, win), color=COLORS['meas'], lw=1.2,
        alpha=0.7, ls='--')
ax.plot(times, sliding_rmse(err_y_imm,  win), color='#7030a0', lw=1.5,
        ls='--', label=f'IMM RMSE_Y (窗口={win}步)')

for ts, te, fc, _ in maneuver_spans:
    ax.axvspan(ts, te, facecolor=fc, alpha=0.6, zorder=0)

ax.set_xlabel('时间 (s)', fontsize=11)
ax.set_ylabel('RMSE (m)', fontsize=11)
ax.set_title('滑动窗口位置 RMSE 对比', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(str(OUT_DIR / 'fig5_sliding_rmse.pdf'), dpi=150, bbox_inches='tight')
fig5.savefig(str(OUT_DIR / 'fig5_sliding_rmse.png'), dpi=150, bbox_inches='tight')
print("Fig 5 saved.")

# ── 打印关键统计信息 ──────────────────────────
print(f"\n{'='*50}")
print("各阶段 IMM 位置 RMSE 统计 (m)")
print(f"{'─'*50}")
segs = [('匀速段 (0-400s)',    0,   400),
        ('慢转弯段 (400-600s)', 400, 600),
        ('快转弯段 (611-660s)', 611, 660),
        ('末段匀速 (660-800s)', 660, 800)]
for name, t0, t1 in segs:
    idx0, idx1 = int(t0/T), int(t1/T)
    rx = np.sqrt(np.mean(err_x_imm[idx0:idx1]**2))
    ry = np.sqrt(np.mean(err_y_imm[idx0:idx1]**2))
    print(f"{name:22s}  X:{rx:7.2f}  Y:{ry:7.2f}")

# 保存数值结果供 LaTeX 引用
stats = {
    'rmse_x_meas': rmse_x_meas, 'rmse_y_meas': rmse_y_meas,
    'rmse_x_imm':  rmse_x_imm,  'rmse_y_imm':  rmse_y_imm,
}
np.save(str(OUT_DIR.parent / 'imm_stats.npy'), stats)
print("\n所有图形已保存。")