"""
Polar Code: encoder + SC decoder + SCL decoder

Reference: Arikan, "Channel Polarization: A Method for Constructing Capacity-Achieving
           Codes for Symmetric Binary-Input Memoryless Channels," IEEE Trans. Inf. Theory, 2009.
"""
import numpy as np


def _bhattacharyya(N, snr_linear):
    """Compute Bhattacharyya parameters for AWGN channel via density evolution."""
    # Initialize: Z(W) for each sub-channel of length 1
    z = np.exp(-snr_linear)  # initial Bhattacharyya param for BPSK-AWGN
    Z = np.array([z])
    for _ in range(int(np.log2(N))):
        # Polarization step
        Z_minus = 2 * Z - Z ** 2   # W-  (worse channel)
        Z_plus = Z ** 2             # W+  (better channel)
        Z = np.zeros(2 * len(Z))
        Z[0::2] = Z_minus
        Z[1::2] = Z_plus
    return Z


def _gaussian_approximation(N, design_snr_db):
    """Polar code channel selection via Gaussian approximation (more accurate)."""
    design_snr = 10 ** (design_snr_db / 10.0)
    sigma2 = 1.0 / (2 * design_snr)

    # m[i] = mean of LLR for sub-channel i, start from channel LLR mean
    m = np.array([2.0 / sigma2])

    def phi(x):
        if x > 10:
            return 1.0
        elif x < 1e-6:
            return 0.0
        return 1 - np.sqrt(np.pi / x) * np.exp(-x / 4) * (1 - 7 / (4 * x))

    def phi_inv(y):
        if y > 1 - 1e-9:
            return 1e10
        elif y < 1e-9:
            return 0.0
        lo, hi = 0.0, 1000.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if phi(mid) < y:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    for _ in range(int(np.log2(N))):
        m_minus = np.array([phi_inv(1 - (1 - phi(mi)) ** 2) for mi in m])
        m_plus = 2 * m
        m_new = np.zeros(2 * len(m))
        m_new[0::2] = m_minus
        m_new[1::2] = m_plus
        m = m_new

    # Higher m => more reliable => info bit; lower m => frozen
    return m


class PolarCode:
    """Polar code with SC and SCL decoding."""

    def __init__(self, N, K, design_snr_db=0.0):
        assert N & (N - 1) == 0, "N must be power of 2"
        assert 0 < K < N
        self.N = N
        self.K = K
        self.n = int(np.log2(N))

        reliability = _gaussian_approximation(N, design_snr_db)
        sorted_idx = np.argsort(reliability)
        self.frozen_idx = set(sorted_idx[:N - K])   # least reliable
        self.info_idx = sorted_idx[N - K:]           # most reliable

        self.G = self._build_generator()

    def _build_generator(self):
        """Build Arikan's polarization matrix G = F^{⊗n} B_N."""
        F = np.array([[1, 0], [1, 1]], dtype=np.int8)
        Fn = F.copy()
        for _ in range(self.n - 1):
            Fn = np.kron(Fn, F)
        # Bit-reversal permutation
        B = self._bit_reversal_permutation()
        return (B @ Fn) % 2

    def _bit_reversal_permutation(self):
        perm = np.arange(self.N)
        for i in range(self.N):
            rev = int(format(i, f'0{self.n}b')[::-1], 2)
            perm[i] = rev
        B = np.zeros((self.N, self.N), dtype=np.int8)
        for i, p in enumerate(perm):
            B[i, p] = 1
        return B

    def encode(self, info_bits):
        """Systematic encoding. Returns N-length codeword."""
        assert len(info_bits) == self.K
        u = np.zeros(self.N, dtype=np.int8)
        u[self.info_idx] = info_bits
        return (self.G.T @ u) % 2  # codeword x = u * G

    # ------------------------------------------------------------------ #
    #  SC Decoder (LLR-based)                                             #
    # ------------------------------------------------------------------ #
    def sc_decode(self, llr):
        """Successive Cancellation decoder. Returns estimated info bits."""
        assert len(llr) == self.N
        est = self._sc_recursive(llr, list(range(self.N)), self.n)
        return est[self.info_idx]

    def _sc_recursive(self, llr, bit_indices, depth):
        if depth == 0:
            idx = bit_indices[0]
            if idx in self.frozen_idx:
                return np.array([0], dtype=np.int8)
            else:
                return np.array([0 if llr[0] >= 0 else 1], dtype=np.int8)

        N = len(llr)
        half = N // 2

        llr_left = llr[:half]
        llr_right = llr[half:]

        # f-function (check node update): min-sum approx
        llr_upper = self._f_func(llr_left, llr_right)

        upper_idx = bit_indices[:half]
        u_upper = self._sc_recursive(llr_upper, upper_idx, depth - 1)

        # g-function: variable node update conditioned on upper decision
        llr_lower = self._g_func(llr_left, llr_right, u_upper)

        lower_idx = bit_indices[half:]
        u_lower = self._sc_recursive(llr_lower, lower_idx, depth - 1)

        u = np.empty(N, dtype=np.int8)
        u[:half] = u_upper
        u[half:] = u_lower
        return u

    @staticmethod
    def _f_func(la, lb):
        """Check-node combining: log-domain min-sum."""
        return np.sign(la) * np.sign(lb) * np.minimum(np.abs(la), np.abs(lb))

    @staticmethod
    def _g_func(la, lb, u):
        """Variable-node combining conditioned on hard decision u."""
        return lb + (1 - 2 * u.astype(float)) * la

    # ------------------------------------------------------------------ #
    #  SCL Decoder                                                         #
    # ------------------------------------------------------------------ #
    def scl_decode(self, llr, L=8):
        """Successive Cancellation List decoder."""
        paths = [{'pm': 0.0, 'bits': np.zeros(self.N, dtype=np.int8)}]

        for i in range(self.N):
            # Compute LLR for position i for each path
            new_paths = []
            for path in paths:
                u_so_far = path['bits'].copy()
                llr_i = self._compute_llr_at_position(llr, u_so_far, i)

                if i in self.frozen_idx:
                    u_so_far[i] = 0
                    pm = path['pm'] + max(0, -llr_i)  # metric for bit=0
                    new_paths.append({'pm': pm, 'bits': u_so_far})
                else:
                    for b in [0, 1]:
                        bits_new = u_so_far.copy()
                        bits_new[i] = b
                        pm = path['pm'] + max(0, (2 * b - 1) * llr_i)
                        new_paths.append({'pm': pm, 'bits': bits_new})

            # Keep L best paths (lowest path metric)
            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:L]

        best = paths[0]['bits']
        return best[self.info_idx]

    def _compute_llr_at_position(self, llr, u_so_far, pos):
        """Compute the LLR at a given bit position given decisions so far."""
        # Simplified: reuse SC scheduling with partial decisions
        # For full SCL, one would maintain a tree; here we recompute via recursion
        return self._sc_llr_recursive(llr, u_so_far, pos, 0, self.N)

    def _sc_llr_recursive(self, llr, u, target, start, length):
        if length == 1:
            return llr[start]

        half = length // 2
        left = llr[start:start + half]
        right = llr[start + half:start + length]

        if target < start + half:
            # Target is in upper half
            alpha_u = self._f_func(left, right)
            return self._sc_llr_recursive(alpha_u, u, target, start, half)
        else:
            # Target is in lower half, need upper decisions
            u_upper_bits = u[start:start + half]
            alpha_l = self._g_func(left, right, u_upper_bits)
            return self._sc_llr_recursive(alpha_l, u, target, start + half, half)
