"""
Transformer-based Neural Polar Decoder.

Architecture:
  - Position embedding + linear projection of LLR inputs
  - Transformer encoder layers (multi-head self-attention + FFN)
  - Linear output head mapping to K decoded bits

Reference concept: Lu et al., "Transformer-based Soft Decoding of Polar Codes,"
arXiv:2002.xxxxx, 2020 (and related work).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerPolarDecoder(nn.Module):
    """
    Transformer decoder for Polar codes.
    Input:  N received LLR values  →  shape (batch, N)
    Output: K estimated info bits  →  shape (batch, K), values in (0,1) via sigmoid
    """

    def __init__(self, N, K, d_model=64, nhead=4, num_layers=3, dim_ff=256, dropout=0.1):
        super().__init__()
        self.N = N
        self.K = K
        self.d_model = d_model

        # Project each LLR scalar to d_model via 1-D token embedding
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=N)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global average pooling then classify K bits
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, K),
        )

    def forward(self, llr):
        """
        llr: (batch, N) tensor of channel LLR values
        returns: (batch, K) sigmoid probabilities
        """
        # (batch, N) -> (batch, N, 1) -> (batch, N, d_model)
        x = self.input_proj(llr.unsqueeze(-1))
        x = self.pos_enc(x)
        x = self.transformer(x)              # (batch, N, d_model)
        x = x.mean(dim=1)                    # global average pool
        logits = self.output_head(x)         # (batch, K)
        return torch.sigmoid(logits)

    def decode(self, llr_np):
        """Hard-decision decode a numpy LLR array of shape (N,). Returns (K,) bits."""
        self.eval()
        with torch.no_grad():
            t = torch.tensor(llr_np, dtype=torch.float32).unsqueeze(0)
            prob = self.forward(t)
            return (prob.squeeze(0) > 0.5).numpy().astype(np.int8)


# ------------------------------------------------------------------ #
#  Training utilities                                                  #
# ------------------------------------------------------------------ #

def generate_training_data(polar_code, n_samples, snr_db, device='cpu'):
    """
    Generate (llr, info_bits) pairs for training.
    Returns tensors on `device`.
    """
    from utils import bpsk_modulate, awgn_channel
    N, K = polar_code.N, polar_code.K
    llrs_all = []
    bits_all = []

    for _ in range(n_samples):
        info = np.random.randint(0, 2, K).astype(np.int8)
        cw = polar_code.encode(info)
        sym = bpsk_modulate(cw)
        _, llr = awgn_channel(sym, snr_db)
        llrs_all.append(llr)
        bits_all.append(info)

    X = torch.tensor(np.array(llrs_all, dtype=np.float32), device=device)
    Y = torch.tensor(np.array(bits_all, dtype=np.float32), device=device)
    return X, Y


def train_transformer_decoder(
    model, polar_code, train_snr_db=2.0, n_train=20000,
    n_val=2000, epochs=30, batch_size=256, lr=1e-3, device='cpu'
):
    """Train the Transformer decoder. Returns list of (train_loss, val_loss) per epoch."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    print(f"Generating {n_train} training samples at SNR={train_snr_db} dB...")
    X_train, Y_train = generate_training_data(polar_code, n_train, train_snr_db, device)
    X_val, Y_val = generate_training_data(polar_code, n_val, train_snr_db, device)

    train_ds = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val).item()

        scheduler.step()
        history.append((train_loss, val_loss))

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    return history
