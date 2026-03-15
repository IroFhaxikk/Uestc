"""
channel_estimation(matlab).py
OFDM LS/DFT Channel Estimation with linear/spline interpolation and MMSE.
Python translation of channel_estimation.m
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ─── System Parameters ────────────────────────────────────────────────────────
Nfft = 512
Ng   = Nfft // 8          # Cyclic prefix length
Nofdm = Nfft + Ng         # Total OFDM symbol length (with CP)
Nsym = 100                # Number of OFDM symbols to simulate
Nps  = 32                 # Pilot spacing
Np   = Nfft // Nps        # Number of pilots per symbol
Nd   = Nfft - Np          # Number of data subcarriers per symbol
Nbps = 4                  # Bits per QAM symbol
M    = 2 ** Nbps          # QAM order (16-QAM)
Es   = 1.0
A    = np.sqrt(3 / 2 / (M - 1) * Es)   # QAM normalization factor
SNRs = [30]               # SNR values to simulate [dB]


# ─── QAM Modulation / Demodulation ───────────────────────────────────────────
def qam_modulate(data: np.ndarray, M: int) -> np.ndarray:
    """Natural-order M-QAM modulation. Maps integers [0, M) to complex symbols."""
    sqrtM = int(np.sqrt(M))
    real_part = (2 * (data // sqrtM) - (sqrtM - 1)).astype(float)
    imag_part = (2 * (data  % sqrtM) - (sqrtM - 1)).astype(float)
    return real_part + 1j * imag_part


def qam_demodulate(symbols: np.ndarray, M: int) -> np.ndarray:
    """Nearest-neighbour M-QAM demodulation. Returns integers [0, M)."""
    sqrtM  = int(np.sqrt(M))
    levels = np.arange(-(sqrtM - 1), sqrtM, 2, dtype=float)
    real_idx = np.argmin(np.abs(symbols.real[:, None] - levels[None, :]), axis=1)
    imag_idx = np.argmin(np.abs(symbols.imag[:, None] - levels[None, :]), axis=1)
    return real_idx * sqrtM + imag_idx


# ─── AWGN ─────────────────────────────────────────────────────────────────────
def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add AWGN noise to a complex signal; SNR is measured from signal power."""
    sig_pow   = np.mean(np.abs(signal) ** 2)
    noise_pow = sig_pow * 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_pow / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


# ─── LS Channel Estimation ────────────────────────────────────────────────────
def LS_CE(Y: np.ndarray, Xp: np.ndarray, pilot_loc: list,
          Nfft: int, Nps: int, int_opt: str) -> np.ndarray:
    """
    LS channel estimation with interpolation.

    Parameters
    ----------
    Y         : Frequency-domain received signal (length Nfft)
    Xp        : Pilot symbols (length Np)
    pilot_loc : Pilot subcarrier indices (0-based)
    Nfft      : FFT size
    Nps       : Pilot spacing
    int_opt   : 'linear' or 'spline'

    Returns
    -------
    H_LS : LS channel estimate at all Nfft subcarriers
    """
    pilot_loc = np.array(pilot_loc)
    LS_est    = Y[pilot_loc] / Xp                     # LS estimate at pilots
    all_loc   = np.arange(Nfft)
    method    = 'linear' if int_opt.lower().startswith('l') else 'cubic'
    f = interp1d(pilot_loc, LS_est, kind=method, fill_value='extrapolate')
    return f(all_loc)


# ─── MMSE Channel Estimation ──────────────────────────────────────────────────
def MMSE_CE(Y: np.ndarray, Xp: np.ndarray, pilot_loc: list,
            Nfft: int, Nps: int, h: np.ndarray, SNR: float) -> np.ndarray:
    """
    MMSE channel estimation based on channel statistics.

    Parameters
    ----------
    Y         : Frequency-domain received signal (length Nfft)
    Xp        : Pilot symbols (length Np)
    pilot_loc : Pilot subcarrier indices (0-based)
    Nfft      : FFT size
    Nps       : Pilot spacing
    h         : True channel impulse response
    SNR       : Signal-to-noise ratio [dB]

    Returns
    -------
    H_MMSE : MMSE channel estimate at all Nfft subcarriers
    """
    snr       = 10 ** (SNR * 0.1)
    Np        = Nfft // Nps
    pilot_loc = np.array(pilot_loc)

    H_tilde = Y[pilot_loc] / Xp      # LS estimate at pilots

    # RMS delay spread
    k_h = np.arange(len(h), dtype=float)
    hh  = h @ h.conj()
    tmp = h * h.conj() * k_h
    r       = np.sum(tmp) / hh
    r2      = (tmp @ k_h) / hh
    tau_rms = np.sqrt(r2 - r ** 2)

    df             = 1.0 / Nfft
    j2pi_tau_df    = 1j * 2 * np.pi * tau_rms * df

    # Cross-correlation matrix: all subcarriers × pilots  (Nfft × Np)
    K1  = np.tile(np.arange(Nfft)[:, None], (1, Np))
    K2  = np.tile(np.arange(Np)[None, :],   (Nfft, 1))
    Rhp = 1.0 / (1 + j2pi_tau_df * (K1 - K2 * Nps))

    # Auto-correlation matrix: pilots × pilots  (Np × Np)
    K3  = np.tile(np.arange(Np)[:, None], (1, Np))
    K4  = np.tile(np.arange(Np)[None, :], (Np, 1))
    Rpp = 1.0 / (1 + j2pi_tau_df * Nps * (K3 - K4)) + np.eye(Np) / snr

    H_MMSE = Rhp @ np.linalg.inv(Rpp) @ H_tilde
    return H_MMSE


# ─── Main Simulation ──────────────────────────────────────────────────────────
np.random.seed(1)

MSEs_all = np.zeros((len(SNRs), 6))

# Pre-create figures so axes persist across the symbol loop
fig1, axes1 = plt.subplots(3, 2, figsize=(12, 9))
fig1.suptitle('Channel Estimation – Power Spectrum')
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
fig2.suptitle('Constellation Diagram')
axes2[0].set_title('Before Equalization')
axes2[1].set_title('After Equalization (MMSE)')
for ax in axes2:
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_aspect('equal')
    ax.grid(True)

method_names = ['LS-linear', 'LS-spline', 'MMSE']

for i, SNR in enumerate(SNRs):
    np.random.seed(1)            # match MATLAB rand/randn seed reset per SNR
    MSE  = np.zeros(6)
    nose = 0

    for nsym in range(Nsym):

        # ── Pilot generation (BPSK ±1) ──────────────────────────────────────
        Xp = (2 * (np.random.randn(Np) > 0) - 1).astype(float)

        # ── Data generation (M-QAM) ─────────────────────────────────────────
        msgint = np.random.randint(0, M, Nd)
        Data   = qam_modulate(msgint, M) * A

        # ── Multiplex pilots and data onto OFDM subcarriers ─────────────────
        X         = np.zeros(Nfft, dtype=complex)
        pilot_loc = []
        ip        = 0
        for k in range(Nfft):
            if k % Nps == 0:
                X[k] = Xp[k // Nps]
                pilot_loc.append(k)
                ip += 1
            else:
                X[k] = Data[k - ip]

        # ── IFFT + Cyclic Prefix ─────────────────────────────────────────────
        x  = np.fft.ifft(X, Nfft)
        xt = np.concatenate([x[Nfft - Ng:], x])      # prepend CP

        # ── 2-tap random channel ─────────────────────────────────────────────
        h = np.array([
            np.random.randn() + 1j * np.random.randn(),
            (np.random.randn() + 1j * np.random.randn()) / 2.0
        ])
        H              = np.fft.fft(h, Nfft)
        channel_length = len(h)
        H_power_dB     = 10 * np.log10(np.abs(H) ** 2 + 1e-12)

        # ── Channel convolution + AWGN ───────────────────────────────────────
        y_channel = np.convolve(xt, h)
        yt        = awgn(y_channel, SNR)

        # ── Remove CP + FFT ──────────────────────────────────────────────────
        y = yt[Ng:Nofdm]
        Y = np.fft.fft(y)

        # ── Channel Estimation (3 methods) ───────────────────────────────────
        H_est_final = None
        for m in range(3):
            if m == 0:
                H_est  = LS_CE(Y, Xp, pilot_loc, Nfft, Nps, 'linear')
            elif m == 1:
                H_est  = LS_CE(Y, Xp, pilot_loc, Nfft, Nps, 'spline')
            else:
                H_est  = MMSE_CE(Y, Xp, pilot_loc, Nfft, Nps, h, SNR)
                H_est_final = H_est          # used for equalization below

            H_est_power_dB = 10 * np.log10(np.abs(H_est) ** 2 + 1e-12)

            # DFT-based channel estimation (noise reduction via IDFT truncation)
            h_est         = np.fft.ifft(H_est)
            h_DFT         = h_est[:channel_length]
            H_DFT         = np.fft.fft(h_DFT, Nfft)
            H_DFT_power_dB = 10 * np.log10(np.abs(H_DFT) ** 2 + 1e-12)

            # ── Plot channel power (first symbol only) ───────────────────────
            if nsym == 0:
                ax_est = axes1[m, 0]
                ax_dft = axes1[m, 1]

                ax_est.plot(H_power_dB,     'b',   linewidth=1, label='True Channel')
                ax_est.plot(H_est_power_dB, 'r:+', markersize=4, linewidth=1,
                            label=method_names[m])
                ax_est.grid(True)
                ax_est.set_title(method_names[m])
                ax_est.set_xlabel('Subcarrier Index')
                ax_est.set_ylabel('Power [dB]')
                ax_est.legend(loc='lower right', fontsize=8)

                ax_dft.plot(H_power_dB,     'b',   linewidth=1, label='True Channel')
                ax_dft.plot(H_DFT_power_dB, 'r:+', markersize=4, linewidth=1,
                            label=f'{method_names[m]} with DFT')
                ax_dft.grid(True)
                ax_dft.set_title(f'{method_names[m]} with DFT')
                ax_dft.set_xlabel('Subcarrier Index')
                ax_dft.set_ylabel('Power [dB]')
                ax_dft.legend(loc='lower right', fontsize=8)

            # ── Accumulate MSE ───────────────────────────────────────────────
            MSE[m]     += np.sum(np.abs(H - H_est) ** 2)
            MSE[m + 3] += np.sum(np.abs(H - H_DFT) ** 2)

        # ── Equalization (MMSE estimate) ─────────────────────────────────────
        Y_eq = Y / H_est_final

        # ── Constellation plot (last 10 symbols) ────────────────────────────
        if nsym >= Nsym - 10:
            axes2[0].plot(Y.real,     Y.imag,     '.', markersize=4, color='C0', alpha=0.6)
            axes2[1].plot(Y_eq.real,  Y_eq.imag,  '.', markersize=4, color='C1', alpha=0.6)

        # ── Data extraction ──────────────────────────────────────────────────
        ip            = 0
        data_idx      = 0
        Data_extracted = np.zeros(Nd, dtype=complex)
        for k in range(Nfft):
            if k % Nps == 0:
                ip += 1
            else:
                Data_extracted[data_idx] = Y_eq[k]
                data_idx += 1

        # ── Demodulation & symbol error count ───────────────────────────────
        msg_detected = qam_demodulate(Data_extracted / A, M)
        nose += int(np.sum(msg_detected != msgint))

    MSEs_all[i, :] = MSE / (Nfft * Nsym)

# ─── Print Results ────────────────────────────────────────────────────────────
print(f'Number of symbol errors = {nose}')
print('MSE of LS-linear/LS-spline/MMSE Channel Estimation = '
      f'{MSEs_all[-1,0]:.4e}/{MSEs_all[-1,1]:.4e}/{MSEs_all[-1,2]:.4e}')
print('MSE of LS-linear/LS-spline/MMSE Channel Estimation with DFT = '
      f'{MSEs_all[-1,3]:.4e}/{MSEs_all[-1,4]:.4e}/{MSEs_all[-1,5]:.4e}')

# ─── Figure 3: MSE vs SNR ─────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.semilogy(SNRs, MSEs_all[:, 0], '-x', label='LS-linear')
ax3.semilogy(SNRs, MSEs_all[:, 2], '-o', label='MMSE')
ax3.set_xlabel('SNR [dB]')
ax3.set_ylabel('MSE')
ax3.set_title('Channel Estimation MSE vs SNR')
ax3.legend()
ax3.grid(True, which='both')

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
plt.show()
