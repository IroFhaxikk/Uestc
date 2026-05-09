import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
class MMSEChannelEstimation:
    """
    基于导频的MMSE（最小均方误差）信道估计
    
    原理说明：
    1. MMSE利用信道的统计特性（频率相关函数）和噪声统计信息
    2. 基于均方根时延扩展 (tau_rms) 构造相关矩阵
    3. 估计公式: H_MMSE = R_hp * inv(R_pp) * H_tilde
       其中 H_tilde 是导频位置的LS估计
    
    ⚠️ 本代码完全遵循Matlab参考实现的逻辑
    
    相关函数模型：
    r(Δf) = 1 / (1 + j*2*pi*tau_rms*Δf)
    """
    
    def __init__(self, Nfft=64, Nps=4, pilot_value=1.0):
        """
        初始化MMSE信道估计器
        
        参数:
            Nfft: 子载波数量（FFT点数）
            Nps: 导频间隔
            pilot_value: 导频符号值
        """
        self.Nfft = Nfft
        self.Nps = Nps
        self.pilot_value = pilot_value
        
        # 生成导频位置索引（从0开始，对应Matlab的1:Nps:Nfft）
        self.pilot_loc = np.arange(0, Nfft, Nps)
        self.Np = len(self.pilot_loc)
        
        print(f"MMSE信道估计器参数:")
        print(f"  FFT点数 (Nfft): {Nfft}")
        print(f"  导频间隔 (Nps): {Nps}")
        print(f"  导频数量 (Np): {self.Np}")
        print(f"  导频位置: {self.pilot_loc}")
    
    def generate_channel(self, channel_length=8):
        """
        生成多径衰落信道（与Matlab代码一致）
        
        Matlab代码:
        h = (randn(1, 8) + 1j*randn(1, 8)) .* exp(-0.1*(0:7));
        H_true = fft(h, Nfft);
        
        参数:
            channel_length: 信道抽头数（默认8）
            
        返回:
            h: 时域信道冲激响应
            H_true: 频域信道响应
        """
        # 生成复高斯随机信道抽头
        h = (np.random.randn(channel_length) + 1j * np.random.randn(channel_length))
        
        # 指数衰减功率延迟谱
        delays = np.arange(channel_length)
        h = h * np.exp(-0.1 * delays)
        
        # FFT到频域
        H_true = np.fft.fft(h, self.Nfft)
        
        return h, H_true
    
    def add_awgn(self, signal, SNR_dB):
        """
        添加加性高斯白噪声
        
        Matlab代码: Y = awgn(Y_clean, SNR, 'measured');
        
        参数:
            signal: 输入信号
            SNR_dB: 信噪比（dB）
            
        返回:
            noisy_signal: 加噪后的信号
        """
        # 计算信号功率
        signal_power = np.mean(np.abs(signal)**2)
        
        # 根据SNR计算噪声功率
        snr_linear = 10**(SNR_dB / 10)
        noise_power = signal_power / snr_linear
        
        # 生成复高斯噪声
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        )
        
        return signal + noise
    
    def calculate_tau_rms(self, h):
        """
        计算均方根时延扩展 (RMS delay spread)
        
        Matlab代码:
        k_h = 0:length(h)-1;
        hh = h * h';
        tmp = h .* conj(h) .* k_h; 
        r = sum(tmp) / hh;
        r2 = (tmp * k_h') / hh;
        tau_rms = sqrt(r2 - r^2);
        
        参数:
            h: 时域信道冲激响应
            
        返回:
            tau_rms: 均方根时延扩展
        """
        k_h = np.arange(len(h))
        
        # 信道总功率
        hh = np.dot(h, np.conj(h))
        
        # 加权功率
        tmp = h * np.conj(h) * k_h
        
        # 一阶矩：平均时延
        r = np.sum(tmp) / hh
        
        # 二阶矩
        r2 = np.dot(tmp, k_h) / hh
        
        # RMS时延扩展
        tau_rms = np.sqrt(r2 - r**2)
        
        return tau_rms
    
    def construct_correlation_matrices(self, h, SNR_dB):
        """
        构造MMSE所需的相关矩阵
        
        Matlab代码:
        df = 1 / Nft;
        j2pi_tau_df = j * 2 * pi * tau_rms * df;
        
        K1 = repmat([0:Nfft-1].', 1, Np);
        K2 = repmat([0:Np-1], Nfft, 1);
        rf = 1 ./ (1 + j2pi_tau_df * (K1 - K2 * Nps)); 
        
        K3 = repmat([0:Np-1].', 1, Np);
        K4 = repmat([0:Np-1], Np, 1);
        rf2 = 1 ./ (1 + j2pi_tau_df * Nps * (K3 - K4));
        
        Rhp = rf;
        Rpp = rf2 + eye(length(H_tilde)) / snr;
        
        参数:
            h: 时域信道冲激响应
            SNR_dB: 信噪比（dB）
            
        返回:
            Rhp: 所有子载波与导频位置的交叉相关矩阵 (Nfft x Np)
            Rpp: 导频位置的自相关矩阵 + 噪声 (Np x Np)
        """
        # 计算tau_rms
        tau_rms = self.calculate_tau_rms(h)
        
        # 频率间隔
        df = 1.0 / self.Nfft
        
        # 线性SNR
        snr_linear = 10**(SNR_dB / 10)
        
        # j * 2 * pi * tau_rms * df
        j2pi_tau_df = 1j * 2 * np.pi * tau_rms * df
        
        # 构造 Rhp (Nfft x Np)
        # K1: 所有子载波索引 [0, 1, 2, ..., Nfft-1] 重复Np列
        # K2: 导频位置编号 [0, 1, 2, ..., Np-1] 重复Nfft行
        K1 = np.tile(np.arange(self.Nfft).reshape(-1, 1), (1, self.Np))
        K2 = np.tile(np.arange(self.Np).reshape(1, -1), (self.Nfft, 1))
        
        # 频率相关函数: r(Δf) = 1 / (1 + j*2*pi*tau_rms*Δf)
        # Δf = (k1 - k2*Nps) * df
        Rhp = 1.0 / (1 + j2pi_tau_df * (K1 - K2 * self.Nps))
        
        # 构造 Rpp (Np x Np)
        K3 = np.tile(np.arange(self.Np).reshape(-1, 1), (1, self.Np))
        K4 = np.tile(np.arange(self.Np).reshape(1, -1), (self.Np, 1))
        
        # 导频位置之间的频率间隔是 Nps*df
        rf2 = 1.0 / (1 + j2pi_tau_df * self.Nps * (K3 - K4))
        
        # 加上噪声项
        Rpp = rf2 + np.eye(self.Np) / snr_linear
        
        return Rhp, Rpp
    
    def mmse_estimate(self, Y, Xp, h, SNR_dB):
        """
        MMSE信道估计（完全对应Matlab的MMSE_CE函数）
        
        Matlab代码:
        function [H_MMSE] = MMSE_CE(Y,Xp,pilot_loc,Nfft,Nps,h,SNR)
            % 导频处 LS 估计
            H_tilde = Y(1, pilot_loc(k_p)) ./ Xp(k_p); 
            % 构造相关矩阵
            Rhp = rf;
            Rpp = rf2 + eye(length(H_tilde)) / snr;
            % 计算最终估计
            H_MMSE = transpose(Rhp * inv(Rpp) * H_tilde.'); 
        end
        
        参数:
            Y: 接收信号（频域，长度为Nfft）
            Xp: 导频符号（长度为Np）
            h: 时域信道冲激响应（用于计算tau_rms）
            SNR_dB: 信噪比（dB）
            
        返回:
            H_MMSE: MMSE估计的信道响应（长度为Nfft）
        """
        # 1. 在导频位置进行LS估计
        k_p = np.arange(self.Np)
        H_tilde = Y[self.pilot_loc[k_p]] / Xp[k_p]
        
        # 2. 构造相关矩阵
        Rhp, Rpp = self.construct_correlation_matrices(h, SNR_dB)
        
        # 3. MMSE估计
        # H_MMSE = Rhp * inv(Rpp) * H_tilde
        # Matlab: H_MMSE = transpose(Rhp * inv(Rpp) * H_tilde.')
        H_MMSE = Rhp @ np.linalg.inv(Rpp) @ H_tilde.reshape(-1, 1)
        H_MMSE = H_MMSE.flatten()
        
        return H_MMSE
    
    def calculate_mse(self, H_true, H_estimated):
        """
        计算均方误差
        
        Matlab代码: mean(abs(H_true.' - H_MMSE.').^2)
        
        参数:
            H_true: 真实信道
            H_estimated: 估计信道
            
        返回:
            mse: 均方误差
        """
        mse = np.mean(np.abs(H_true - H_estimated)**2)
        return mse


def simulate_mmse_channel_estimation():
    """
    MMSE信道估计仿真（对应Matlab主程序）
    """
    print("="*70)
    print("基于导频的MMSE信道估计仿真（遵循Matlab逻辑）")
    print("="*70)
    
    # 系统参数（与Matlab代码一致）
    Nfft = 64
    Nps = 4
    pilot_value = 1.0
    SNR_dB = 20
    
    # 创建MMSE估计器
    estimator = MMSEChannelEstimation(Nfft, Nps, pilot_value)
    
    # 创建LS估计器用于对比
    from 纠错编码.Code.Traditonal.Code.python.LS import LSChannelEstimation
    ls_estimator = LSChannelEstimation(Nfft, Nps, pilot_value)
    
    # 生成信道
    np.random.seed(42)
    h, H_true = estimator.generate_channel(channel_length=8)
    
    # 计算tau_rms
    tau_rms = estimator.calculate_tau_rms(h)
    
    print(f"\n信道参数:")
    print(f"  时域抽头数: {len(h)}")
    print(f"  RMS时延扩展 (tau_rms): {tau_rms:.4f}")
    print(f"  时域信道: h[:3] = {h[:3]}")
    
    # 构造发送信号
    Xp = np.ones(estimator.Np) * pilot_value  # 导频符号
    X = np.ones(Nfft)  # 所有子载波初始化为1
    X[estimator.pilot_loc] = Xp  # 导频位置
    
    # 接收信号（通过信道）
    Y_clean = H_true * X
    
    # 加噪声
    Y = estimator.add_awgn(Y_clean, SNR_dB)
    
    # MMSE信道估计
    H_MMSE = estimator.mmse_estimate(Y, Xp, h, SNR_dB)
    
    # LS信道估计（用于对比）
    H_LS = ls_estimator.ls_estimate(Y, Xp, int_opt='linear')
    
    # 计算MSE
    mse_mmse = estimator.calculate_mse(H_true, H_MMSE)
    mse_ls = estimator.calculate_mse(H_true, H_LS)
    
    print(f"\n估计性能 (SNR = {SNR_dB} dB):")
    print(f"  MMSE MSE: {mse_mmse:.6f} ({10*np.log10(mse_mmse):6.2f} dB)")
    print(f"  LS   MSE: {mse_ls:.6f} ({10*np.log10(mse_ls):6.2f} dB)")
    print(f"  性能增益: {10*np.log10(mse_ls/mse_mmse):.2f} dB")
    
    # 可视化
    visualize_comparison(H_true, H_MMSE, H_LS, estimator.pilot_loc, Nfft, SNR_dB)
    
    return H_true, H_MMSE, H_LS


def visualize_comparison(H_true, H_MMSE, H_LS, pilot_loc, Nfft, SNR_dB):
    """
    可视化MMSE和LS的对比结果
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    subcarrier_indices = np.arange(Nfft)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 幅度响应
    axes[0, 0].plot(subcarrier_indices, np.abs(H_true), 'b-', 
                   label='真实信道', linewidth=2.5, alpha=0.8)
    axes[0, 0].plot(subcarrier_indices, np.abs(H_MMSE), 'r--', 
                   label='MMSE估计', linewidth=2)
    axes[0, 0].plot(subcarrier_indices, np.abs(H_LS), 'm:', 
                   label='LS估计', linewidth=2)
    axes[0, 0].scatter(pilot_loc, np.abs(H_true[pilot_loc]), 
                      c='green', s=100, marker='o', label='导频位置', zorder=5)
    axes[0, 0].set_xlabel('子载波索引', fontsize=11)
    axes[0, 0].set_ylabel('幅度', fontsize=11)
    axes[0, 0].set_title(f'信道幅度响应对比 (SNR={SNR_dB}dB)', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 相位响应
    axes[0, 1].plot(subcarrier_indices, np.angle(H_true), 'b-', 
                   label='真实信道', linewidth=2.5, alpha=0.8)
    axes[0, 1].plot(subcarrier_indices, np.angle(H_MMSE), 'r--', 
                   label='MMSE估计', linewidth=2)
    axes[0, 1].plot(subcarrier_indices, np.angle(H_LS), 'm:', 
                   label='LS估计', linewidth=2)
    axes[0, 1].scatter(pilot_loc, np.angle(H_true[pilot_loc]), 
                      c='green', s=100, marker='o', label='导频位置', zorder=5)
    axes[0, 1].set_xlabel('子载波索引', fontsize=11)
    axes[0, 1].set_ylabel('相位（弧度）', fontsize=11)
    axes[0, 1].set_title('信道相位响应对比', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 估计误差（幅度）
    error_mmse = np.abs(H_true - H_MMSE)
    error_ls = np.abs(H_true - H_LS)
    
    axes[1, 0].plot(subcarrier_indices, error_mmse, 'r-', 
                   label='MMSE估计', linewidth=2)
    axes[1, 0].plot(subcarrier_indices, error_ls, 'm-', 
                   label='LS估计', linewidth=2)
    axes[1, 0].set_xlabel('子载波索引', fontsize=11)
    axes[1, 0].set_ylabel('估计误差', fontsize=11)
    axes[1, 0].set_title('信道估计误差对比', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 复平面（星座图）
    axes[1, 1].scatter(np.real(H_true), np.imag(H_true), 
                      s=100, c='blue', marker='o', label='真实信道', alpha=0.6)
    axes[1, 1].scatter(np.real(H_MMSE), np.imag(H_MMSE), 
                      s=50, c='red', marker='x', label='MMSE估计', alpha=0.8, linewidths=2)
    axes[1, 1].scatter(np.real(H_LS), np.imag(H_LS), 
                      s=30, c='magenta', marker='.', label='LS估计', alpha=0.6)
    axes[1, 1].set_xlabel('实部', fontsize=11)
    axes[1, 1].set_ylabel('虚部', fontsize=11)
    axes[1, 1].set_title('复平面信道响应', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.show()


def compare_snr_performance():
    """
    比较MMSE和LS在不同SNR下的性能（对应Matlab的蒙特卡洛仿真）
    """
    print("\n" + "="*70)
    print("="*70)
    
    # 参数设置（与Matlab一致）
    Nfft = 64
    Nps = 4
    snr_range = np.arange(0, 31, 5)  # 0:5:30
    num_iter = 100
    
    estimator = MMSEChannelEstimation(Nfft, Nps)
    
    from 纠错编码.Code.Traditonal.Code.python.LS import LSChannelEstimation
    ls_estimator = LSChannelEstimation(Nfft, Nps)
    
    mse_ls = np.zeros(len(snr_range))
    mse_mmse = np.zeros(len(snr_range))
    
    print(f"\n运行 {num_iter} 次蒙特卡洛仿真...")
    
    for s_idx, SNR in enumerate(snr_range):
        err_ls = 0
        err_mmse = 0
        
        for n in range(num_iter):
            # 生成信道
            h, H_true = estimator.generate_channel(channel_length=8)
            
            # 构造发送信号
            Xp = np.ones(estimator.Np)
            X = np.ones(Nfft)
            X[estimator.pilot_loc] = Xp
            
            # 接收信号
            Y_clean = H_true * X
            Y = estimator.add_awgn(Y_clean, SNR)
            
            # LS估计
            H_LS = ls_estimator.ls_estimate(Y, Xp, int_opt='linear')
            
            # MMSE估计
            H_MMSE = estimator.mmse_estimate(Y, Xp, h, SNR)
            
            # 累积误差
            err_ls += np.mean(np.abs(H_true - H_LS)**2)
            err_mmse += np.mean(np.abs(H_true - H_MMSE)**2)
        
        mse_ls[s_idx] = err_ls / num_iter
        mse_mmse[s_idx] = err_mmse / num_iter
        
        gain_db = 10*np.log10(mse_ls[s_idx] / mse_mmse[s_idx])
        
        print(f"SNR = {SNR:2d} dB | "
              f"LS: {10*np.log10(mse_ls[s_idx]):6.2f} dB | "
              f"MMSE: {10*np.log10(mse_mmse[s_idx]):6.2f} dB | "
              f"增益: {gain_db:5.2f} dB")
    
    # 绘制结果（对应Matlab的semilogy图）
    plt.figure(figsize=(10, 7))
    plt.semilogy(snr_range, mse_ls, 'b-o', linewidth=2.5, markersize=8, 
                label='LS 估计 (Linear)')
    plt.semilogy(snr_range, mse_mmse, 'r-s', linewidth=2.5, markersize=8, 
                label='MMSE 估计')
    plt.xlabel('SNR (dB)', fontsize=13)
    plt.ylabel('均方误差 (MSE)', fontsize=13)
    plt.title('LS 与 MMSE 信道估计性能对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return snr_range, mse_ls, mse_mmse


if __name__ == "__main__":
    # 单次仿真
    H_true, H_MMSE, H_LS = simulate_mmse_channel_estimation()
    
    # 性能对比
    snr_range, mse_ls, mse_mmse = compare_snr_performance()
    
    print("\n" + "="*70)
    print("MMSE信道估计原理总结")
    print("="*70)
