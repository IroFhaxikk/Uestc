"""
Main simulation script:
  - Compares SC decoder (Polar), BP decoder (LDPC), and Transformer decoder (Polar)
  - BER vs Eb/N0 curves for different code lengths and code rates
  - Generates plots saved to ../results/
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure local imports work when run from any directory
sys.path.insert(0, os.path.dirname(__file__))

from utils import bpsk_modulate, awgn_channel, compute_ber
from polar_code import PolarCode
from ldpc_code import LDPCCode
from transformer_decoder import TransformerPolarDecoder, train_transformer_decoder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
# Simulation helpers                                                    #
# ------------------------------------------------------------------ #

def simulate_polar_sc(N, K, snr_range, n_frames=500, n_errors_min=100):
    """BER curve for Polar + SC decoder."""
    pc = PolarCode(N, K, design_snr_db=0.0)
    rate = K / N
    bers = []
    for snr_db in snr_range:
        errors = 0
        bits_total = 0
        frames = 0
        while errors < n_errors_min and frames < n_frames:
            info = np.random.randint(0, 2, K).astype(np.int8)
            cw = pc.encode(info)
            sym = bpsk_modulate(cw)
            _, llr = awgn_channel(sym, snr_db)
            dec = pc.sc_decode(llr)
            errors += int(np.sum(info != dec))
            bits_total += K
            frames += 1
        ber = errors / bits_total if bits_total > 0 else 1.0
        bers.append(max(ber, 1e-7))
        print(f"  Polar SC  N={N} K={K} SNR={snr_db:.1f}dB  BER={ber:.4e}  ({frames} frames)")
    return bers


def simulate_ldpc_bp(N, K, snr_range, n_frames=300, n_errors_min=50, max_iter=50):
    """BER curve for LDPC + BP decoder."""
    lc = LDPCCode(N, K, dv=3)
    bers = []
    for snr_db in snr_range:
        errors = 0
        bits_total = 0
        frames = 0
        while errors < n_errors_min and frames < n_frames:
            info = np.random.randint(0, 2, K).astype(np.int8)
            cw = lc.encode(info)
            sym = bpsk_modulate(cw)
            _, llr = awgn_channel(sym, snr_db)
            dec_cw = lc.bp_decode(llr, max_iter=max_iter)
            dec_info = dec_cw[lc.M:]  # systematic part
            errors += int(np.sum(info != dec_info[:K]))
            bits_total += K
            frames += 1
        ber = errors / bits_total if bits_total > 0 else 1.0
        bers.append(max(ber, 1e-7))
        print(f"  LDPC BP   N={N} K={K} SNR={snr_db:.1f}dB  BER={ber:.4e}  ({frames} frames)")
    return bers


def simulate_transformer_polar(model, polar_code, snr_range, n_frames=500, n_errors_min=100):
    """BER curve for Polar code decoded by Transformer model."""
    pc = polar_code
    bers = []
    for snr_db in snr_range:
        errors = 0
        bits_total = 0
        frames = 0
        while errors < n_errors_min and frames < n_frames:
            info = np.random.randint(0, 2, pc.K).astype(np.int8)
            cw = pc.encode(info)
            sym = bpsk_modulate(cw)
            _, llr = awgn_channel(sym, snr_db)
            dec = model.decode(llr)
            errors += int(np.sum(info != dec))
            bits_total += pc.K
            frames += 1
        ber = errors / bits_total if bits_total > 0 else 1.0
        bers.append(max(ber, 1e-7))
        print(f"  TF Polar  N={pc.N} K={pc.K} SNR={snr_db:.1f}dB  BER={ber:.4e}  ({frames} frames)")
    return bers


# ------------------------------------------------------------------ #
# Experiment 1: BER vs SNR for fixed code (N=64, R=1/2)               #
# ------------------------------------------------------------------ #

def experiment_ber_vs_snr():
    print("\n=== Experiment 1: BER vs Eb/N0 (N=64, K=32) ===")
    N, K = 64, 32
    snr_range = np.arange(0, 6.5, 0.5)

    # --- Polar SC ---
    ber_sc = simulate_polar_sc(N, K, snr_range, n_frames=800)

    # --- LDPC BP ---
    ber_bp = simulate_ldpc_bp(N, K, snr_range, n_frames=400, max_iter=50)

    # --- Transformer Polar ---
    pc = PolarCode(N, K, design_snr_db=0.0)
    model = TransformerPolarDecoder(N, K, d_model=64, nhead=4, num_layers=3, dim_ff=128)

    # Try to load cached model
    model_path = os.path.join(RESULTS_DIR, f'tf_polar_N{N}_K{K}.pt')
    if os.path.exists(model_path):
        import torch
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Loaded cached Transformer model.")
    else:
        history = train_transformer_decoder(
            model, pc, train_snr_db=2.5, n_train=30000, n_val=3000,
            epochs=40, batch_size=512
        )
        import torch
        torch.save(model.state_dict(), model_path)
        # Plot training loss
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([h[0] for h in history], label='Train Loss')
        ax.plot([h[1] for h in history], label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BCE Loss')
        ax.set_title('Transformer Decoder Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, 'training_loss.png'), dpi=150)
        plt.close(fig)

    ber_tf = simulate_transformer_polar(model, pc, snr_range, n_frames=800)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snr_range, ber_sc, 'b-o', markersize=5, label='Polar SC (N=64, R=1/2)')
    ax.semilogy(snr_range, ber_bp, 'r-s', markersize=5, label='LDPC BP (N=64, R=1/2)')
    ax.semilogy(snr_range, ber_tf, 'g-^', markersize=5, label='Transformer-Polar (N=64, R=1/2)')
    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('BER vs Eb/N0: Polar SC vs LDPC BP vs Transformer', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-5, 1.0])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ber_vs_snr_N64.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: ber_vs_snr_N64.png")

    return snr_range, ber_sc, ber_bp, ber_tf


# ------------------------------------------------------------------ #
# Experiment 2: BER vs code length (different N at fixed SNR=3dB)     #
# ------------------------------------------------------------------ #

def experiment_ber_vs_codelength():
    print("\n=== Experiment 2: BER vs Code Length (R=1/2, SNR=3dB) ===")
    snr_fixed = 3.0
    N_values = [32, 64, 128, 256]
    rates = [0.5] * len(N_values)

    ber_sc_list, ber_bp_list = [], []
    for N in N_values:
        K = N // 2
        snr_range = [snr_fixed]
        ber_sc = simulate_polar_sc(N, K, snr_range, n_frames=1000)
        ber_bp = simulate_ldpc_bp(N, K, snr_range, n_frames=500, max_iter=50)
        ber_sc_list.append(ber_sc[0])
        ber_bp_list.append(ber_bp[0])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(N_values, ber_sc_list, 'b-o', markersize=6, label='Polar SC')
    ax.semilogy(N_values, ber_bp_list, 'r-s', markersize=6, label='LDPC BP')
    ax.set_xlabel('Code Length N', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title(f'BER vs Code Length (R=1/2, Eb/N0={snr_fixed}dB)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(N_values)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ber_vs_codelength.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: ber_vs_codelength.png")


# ------------------------------------------------------------------ #
# Experiment 3: BER vs code rate (N=64, SNR=3dB)                      #
# ------------------------------------------------------------------ #

def experiment_ber_vs_rate():
    print("\n=== Experiment 3: BER vs Code Rate (N=64, SNR=3dB) ===")
    N = 64
    snr_fixed = 3.0
    K_values = [16, 24, 32, 40, 48]
    rates = [k / N for k in K_values]

    ber_sc_list, ber_bp_list = [], []
    for K in K_values:
        snr_range = [snr_fixed]
        ber_sc = simulate_polar_sc(N, K, snr_range, n_frames=1000)
        ber_bp = simulate_ldpc_bp(N, K, snr_range, n_frames=500, max_iter=50)
        ber_sc_list.append(ber_sc[0])
        ber_bp_list.append(ber_bp[0])
        print(f"  N={N} K={K} rate={K/N:.3f}: SC={ber_sc[0]:.4e}  BP={ber_bp[0]:.4e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(rates, ber_sc_list, 'b-o', markersize=6, label='Polar SC')
    ax.semilogy(rates, ber_bp_list, 'r-s', markersize=6, label='LDPC BP')
    ax.set_xlabel('Code Rate R = K/N', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title(f'BER vs Code Rate (N={N}, Eb/N0={snr_fixed}dB)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ber_vs_rate.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: ber_vs_rate.png")


# ------------------------------------------------------------------ #
# Experiment 4: Transformer detailed BER vs SNR (multiple SNRs)       #
# ------------------------------------------------------------------ #

def experiment_transformer_detail():
    print("\n=== Experiment 4: Transformer Polar Decoder BER Curve (N=64, R=1/2) ===")
    N, K = 64, 32
    snr_range = np.arange(-1, 7.0, 1.0)

    pc = PolarCode(N, K, design_snr_db=0.0)
    model = TransformerPolarDecoder(N, K, d_model=64, nhead=4, num_layers=3, dim_ff=128)

    model_path = os.path.join(RESULTS_DIR, f'tf_polar_N{N}_K{K}.pt')
    if os.path.exists(model_path):
        import torch
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Loaded cached Transformer model.")
    else:
        train_transformer_decoder(
            model, pc, train_snr_db=2.5, n_train=30000, n_val=3000,
            epochs=40, batch_size=512
        )
        import torch
        torch.save(model.state_dict(), model_path)

    ber_sc = simulate_polar_sc(N, K, snr_range, n_frames=1000)
    ber_tf = simulate_transformer_polar(model, pc, snr_range, n_frames=1000)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snr_range, ber_sc, 'b-o', markersize=5, label='Polar SC Decoder')
    ax.semilogy(snr_range, ber_tf, 'g-^', markersize=5, label='Transformer Decoder')
    ax.set_xlabel('Eb/N0 (dB)', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('Transformer vs SC Decoder for Polar Codes (N=64, R=1/2)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-5, 1.0])
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'ber_transformer_vs_sc.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: ber_transformer_vs_sc.png")


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    np.random.seed(42)
    t0 = time.time()

    print("=" * 60)
    print("5G信道编译码仿真 — Polar码 vs LDPC码 vs Transformer解码器")
    print("=" * 60)

    experiment_ber_vs_snr()
    experiment_ber_vs_codelength()
    experiment_ber_vs_rate()
    experiment_transformer_detail()

    elapsed = time.time() - t0
    print(f"\n所有仿真完成，耗时 {elapsed:.1f} 秒。结果保存于 results/ 目录。")
