import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class LSChannelEstimation:
    """
    基于导频的LS（最小二乘）信道估计
    
    原理说明：
    1. LS信道估计是最简单的信道估计方法之一
    2. 在OFDM系统中，在已知的导频位置插入导频符号
    3. 接收端通过 H_LS = Y / X 来估计信道（Y是接收信号，X是发送导频）
    4. 对于非导频位置，使用插值方法获得完整的信道估计
    
    ⚠️ 本代码完全遵循Matlab参考实现的逻辑
    """
    
    def __init__(self, Nfft=64, Nps=4, pilot_value=1.0):
        """
        初始化LS信道估计器
        
        参数:
            Nfft: 子载波数量（FFT点数）
            Nps: 导频间隔（每隔Nps个子载波插入一个导频）
            pilot_value: 导频符号值（默认为1）
        """
        self.Nfft = Nfft
        self.Nps = Nps
        self.pilot_value = pilot_value
        
        # 生成导频位置索引（从0开始，对应Matlab的1:Nps:Nfft）
        self.pilot_loc = np.arange(0, Nfft, Nps)
        self.Np = len(self.pilot_loc)
        
        print(f"LS信道估计器参数:")
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
    
    def ls_estimate(self, Y, Xp, int_opt='linear'):
        """
        LS信道估计（完全对应Matlab的LS_CE函数）
        
        Matlab代码:
        function [H_LS] = LS_CE(Y,Xp,pilot_loc,Nfft,Nps,int_opt)
            Np = Nfft/Nps;
            % 1. 导频处 LS 估计
            LS_est = Y(pilot_loc) ./ Xp;
            % 2. 插值
            all_indices = 1:Nfft;
            H_LS = interp1(pilot_loc, LS_est, all_indices, method, 'extrap');
        end
        
        参数:
            Y: 接收信号（频域，长度为Nfft）
            Xp: 导频符号（长度为Np）
            int_opt: 插值方法 ('linear' 或 'spline')
            
        返回:
            H_LS: LS估计的信道响应（长度为Nfft）
        """
        # 1. 在导频位置进行LS估计
        LS_est = Y[self.pilot_loc] / Xp
        
        # 2. 确定插值方法
        if int_opt.lower() in ['1', 'linear']:
            method = 'linear'
        else:
            method = 'cubic'  # scipy的'cubic'对应Matlab的'spline'
        
        # 3. 插值到所有子载波
        # Matlab: all_indices = 1:Nfft (从1开始)
        # Python: all_indices = 0:(Nfft-1) (从0开始)
        all_indices = np.arange(self.Nfft)
        
        # 分别对实部和虚部插值
        interp_func_real = interp1d(self.pilot_loc, np.real(LS_est), 
                                    kind=method, fill_value='extrapolate')
        interp_func_imag = interp1d(self.pilot_loc, np.imag(LS_est), 
                                    kind=method, fill_value='extrapolate')
        
        H_LS = interp_func_real(all_indices) + 1j * interp_func_imag(all_indices)
        
        return H_LS
    
    def calculate_mse(self, H_true, H_estimated):
        """
        计算均方误差
        
        Matlab代码: mean(abs(H_true.' - H_LS.').^2)
        
        参数:
            H_true: 真实信道
            H_estimated: 估计信道
            
        返回:
            mse: 均方误差
        """
        mse = np.mean(np.abs(H_true - H_estimated)**2)
        return mse


def simulate_ls_channel_estimation():
    """
    LS信道估计仿真（对应Matlab主程序）
    """
    print("="*70)
    print("基于导频的LS信道估计仿真（遵循Matlab逻辑）")
    print("="*70)
    
    # 系统参数（与Matlab代码一致）
    Nfft = 64
    Nps = 4
    pilot_value = 1.0
    SNR_dB = 20
    
    # 创建LS估计器
    estimator = LSChannelEstimation(Nfft, Nps, pilot_value)
    
    # 生成信道
    np.random.seed(42)
    h, H_true = estimator.generate_channel(channel_length=8)
    
    print(f"\n信道参数:")
    print(f"  时域抽头数: {len(h)}")
    print(f"  时域信道: h[:3] = {h[:3]}")
    
    # 构造发送信号
    Xp = np.ones(estimator.Np) * pilot_value  # 导频符号
    X = np.ones(Nfft)  # 所有子载波初始化为1
    X[estimator.pilot_loc] = Xp  # 导频位置
    
    # 接收信号（通过信道）
    Y_clean = H_true * X
    
    # 加噪声
    Y = estimator.add_awgn(Y_clean, SNR_dB)
    
    # LS信道估计
    H_LS = estimator.ls_estimate(Y, Xp, int_opt='linear')
    
    # 计算MSE
    mse = estimator.calculate_mse(H_true, H_LS)
    
    print(f"\n估计性能 (SNR = {SNR_dB} dB):")
    print(f"  MSE: {mse:.6f}")
    print(f"  MSE (dB): {10*np.log10(mse):.2f} dB")
    
    # 可视化
    visualize_single_result(H_true, H_LS, estimator.pilot_loc, Nfft, SNR_dB)
    
    return H_true, H_LS, mse


def visualize_single_result(H_true, H_LS, pilot_loc, Nfft, SNR_dB):
    """
    可视化单次仿真结果
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    subcarrier_indices = np.arange(Nfft)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # 幅度响应
    axes[0, 0].plot(subcarrier_indices, np.abs(H_true), 'b-', 
                   label='真实信道', linewidth=2.5, alpha=0.8)
    axes[0, 0].plot(subcarrier_indices, np.abs(H_LS), 'r--', 
                   label='LS估计', linewidth=2)
    axes[0, 0].scatter(pilot_loc, np.abs(H_true[pilot_loc]), 
                      c='green', s=100, marker='o', label='导频位置', zorder=5)
    axes[0, 0].set_xlabel('子载波索引')
    axes[0, 0].set_ylabel('幅度')
    axes[0, 0].set_title(f'信道幅度响应 (SNR={SNR_dB}dB)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 相位响应
    axes[0, 1].plot(subcarrier_indices, np.angle(H_true), 'b-', 
                   label='真实信道', linewidth=2.5, alpha=0.8)
    axes[0, 1].plot(subcarrier_indices, np.angle(H_LS), 'r--', 
                   label='LS估计', linewidth=2)
    axes[0, 1].scatter(pilot_loc, np.angle(H_true[pilot_loc]), 
                      c='green', s=100, marker='o', label='导频位置', zorder=5)
    axes[0, 1].set_xlabel('子载波索引')
    axes[0, 1].set_ylabel('相位（弧度）')
    axes[0, 1].set_title('信道相位响应')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 实部
    axes[1, 0].plot(subcarrier_indices, np.real(H_true), 'b-', 
                   label='真实信道', linewidth=2.5, alpha=0.8)
    axes[1, 0].plot(subcarrier_indices, np.real(H_LS), 'r--', 
                   label='LS估计', linewidth=2)
    axes[1, 0].scatter(pilot_loc, np.real(H_true[pilot_loc]), 
                      c='green', s=100, marker='o', label='导频位置', zorder=5)
    axes[1, 0].set_xlabel('子载波索引')
    axes[1, 0].set_ylabel('实部')
    axes[1, 0].set_title('信道响应实部')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 虚部
    axes[1, 1].plot(subcarrier_indices, np.imag(H_true), 'b-', 
                   label='真实信道', linewidth=2.5, alpha=0.8)
    axes[1, 1].plot(subcarrier_indices, np.imag(H_LS), 'r--', 
                   label='LS估计', linewidth=2)
    axes[1, 1].scatter(pilot_loc, np.imag(H_true[pilot_loc]), 
                      c='green', s=100, marker='o', label='导频位置', zorder=5)
    axes[1, 1].set_xlabel('子载波索引')
    axes[1, 1].set_ylabel('虚部')
    axes[1, 1].set_title('信道响应虚部')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_snr_performance():
    """
    比较不同SNR下的LS信道估计性能（对应Matlab的蒙特卡洛仿真）
    """
    print("\n" + "="*70)
    print("不同SNR下的LS信道估计性能（蒙特卡洛仿真）")
    print("="*70)
    
    # 参数设置（与Matlab一致）
    Nfft = 64
    Nps = 4
    snr_range = np.arange(0, 31, 5)  # 0:5:30
    num_iter = 100
    
    estimator = LSChannelEstimation(Nfft, Nps)
    
    mse_ls = np.zeros(len(snr_range))
    
    print(f"\n运行 {num_iter} 次蒙特卡洛仿真...")
    
    for s_idx, SNR in enumerate(snr_range):
        err_ls = 0
        
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
            H_LS = estimator.ls_estimate(Y, Xp, int_opt='linear')
            
            # 累积误差
            err_ls += np.mean(np.abs(H_true - H_LS)**2)
        
        mse_ls[s_idx] = err_ls / num_iter
        print(f"SNR = {SNR:2d} dB | MSE = {mse_ls[s_idx]:.6f} | "
              f"MSE(dB) = {10*np.log10(mse_ls[s_idx]):6.2f} dB")
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, mse_ls, 'b-o', linewidth=2.5, markersize=8, 
                label='LS估计 (Linear)')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('均方误差 (MSE)', fontsize=12)
    plt.title('LS信道估计性能 vs SNR', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return snr_range, mse_ls


if __name__ == "__main__":
    # 单次仿真
    H_true, H_LS, mse = simulate_ls_channel_estimation()
    # 性能对比
    snr_range, mse_ls = compare_snr_performance()


