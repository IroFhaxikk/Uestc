"""
Utility functions: BPSK modulation, AWGN channel, BER calculation
"""
import numpy as np


def bpsk_modulate(bits):
    """Map bits to BPSK symbols: 0 -> +1, 1 -> -1"""
    return 1 - 2 * bits.astype(float)


def awgn_channel(symbols, snr_db):
    """Add AWGN noise; returns received signal and LLR values."""
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std = 1.0 / np.sqrt(2 * snr_linear)
    noise = np.random.randn(*symbols.shape) * noise_std
    received = symbols + noise
    # LLR = 2*y/sigma^2  (for BPSK, x in {+1,-1})
    llr = 2 * received / (noise_std ** 2)
    return received, llr


def compute_ber(tx_bits, rx_bits):
    return np.sum(tx_bits != rx_bits) / len(tx_bits)


def snr_db_to_noise_std(snr_db, rate=1.0):
    """Convert Eb/N0 (dB) to noise std, accounting for code rate."""
    snr_linear = 10 ** (snr_db / 10.0)
    # Es/N0 = (Eb/N0) * rate  for BPSK Es=1
    noise_var = 1.0 / (2 * snr_linear * rate)
    return np.sqrt(noise_var)
