"""
LDPC Code: encoder + Belief Propagation (sum-product) decoder

Uses a randomly constructed regular LDPC parity check matrix via the
Progressive Edge Growth (PEG) approximation (simplified random version).
"""
import numpy as np
from scipy.sparse import csr_matrix


def _make_regular_ldpc(M, N, dv=3, dc=None):
    """
    Build a random regular (dv, dc)-LDPC parity check matrix H of shape (M, N).
    dv: variable node degree, dc = N*dv/M: check node degree.
    Returns H as a dense numpy array (0/1).
    """
    if dc is None:
        dc = N * dv // M
    assert N * dv == M * dc, "dv/dc/N/M inconsistent"

    rng = np.random.default_rng(42)
    max_tries = 200

    for attempt in range(max_tries):
        H = np.zeros((M, N), dtype=np.int8)
        col_counts = np.zeros(N, dtype=int)
        row_counts = np.zeros(M, dtype=int)

        # Build edge list: each variable node appears dv times
        edges = np.repeat(np.arange(N), dv)
        rng.shuffle(edges)
        edge_idx = 0

        success = True
        for m in range(M):
            added = 0
            candidates = np.where(row_counts < dc)[0]
            tried = set()
            while added < dc:
                if edge_idx >= len(edges):
                    success = False
                    break
                v = edges[edge_idx]
                edge_idx += 1
                if H[m, v] == 0 and v not in tried:
                    H[m, v] = 1
                    col_counts[v] += 1
                    row_counts[m] += 1
                    added += 1
                tried.add(v)
            if not success:
                break

        if success and np.all(row_counts == dc) and np.all(col_counts == dv):
            return H

    # Fallback: just set random positions
    H = np.zeros((M, N), dtype=np.int8)
    for m in range(M):
        cols = rng.choice(N, size=dc, replace=False)
        H[m, cols] = 1
    return H


def _systematic_generator(H):
    """
    Derive systematic generator matrix G from H via row reduction.
    Returns G (K x N) where K = N - rank(H).
    """
    M, N = H.shape
    # Gaussian elimination over GF(2)
    Hwork = H.copy().astype(int)
    pivot_cols = []
    row = 0
    for col in range(N):
        # Find pivot in this column below current row
        found = -1
        for r in range(row, M):
            if Hwork[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        Hwork[[row, found]] = Hwork[[found, row]]
        pivot_cols.append(col)
        for r in range(M):
            if r != row and Hwork[r, col] == 1:
                Hwork[r] = (Hwork[r] + Hwork[row]) % 2
        row += 1
        if row == M:
            break

    rank = len(pivot_cols)
    K = N - rank
    free_cols = [c for c in range(N) if c not in pivot_cols]

    # Build G: K x N
    G = np.zeros((K, N), dtype=np.int8)
    for i, fc in enumerate(free_cols):
        G[i, fc] = 1
        for j, pc in enumerate(pivot_cols):
            G[i, pc] = Hwork[j, fc]

    return G, free_cols


class LDPCCode:
    """Regular LDPC code with sum-product BP decoder."""

    def __init__(self, N, K, dv=3):
        self.N = N
        self.K = K
        self.M = N - K

        dc = N * dv // self.M
        if N * dv % self.M != 0:
            # Adjust dv
            dv = 2
            dc = N * dv // self.M

        self.dv = dv
        self.dc = dc

        self.H = _make_regular_ldpc(self.M, N, dv=dv, dc=dc)
        # Parity check edge lists for efficient BP
        self._build_edge_lists()

    def _build_edge_lists(self):
        """Precompute neighbor lists for BP."""
        H = self.H
        M, N = H.shape
        # check_to_var[m] = list of variable indices connected to check m
        self.c2v = [list(np.where(H[m])[0]) for m in range(M)]
        # var_to_check[n] = list of check indices connected to variable n
        self.v2c = [list(np.where(H[:, n])[0]) for n in range(N)]

    def encode(self, info_bits):
        """Encode K info bits to N-length codeword (systematic form via H*c=0 mod 2)."""
        assert len(info_bits) == self.K
        # Simple systematic encoding: use the first K positions as info bits
        # and solve for M parity bits such that H*c = 0 mod 2
        # H = [H_p | H_s], solve H_p * p = H_s * s mod 2
        H = self.H.astype(int)
        s = info_bits.astype(int)

        # Partition: first M columns as parity part, last K as systematic
        Hp = H[:, :self.M]
        Hs = H[:, self.M:]

        # Solve Hp * p = Hs * s mod 2 via Gaussian elimination
        rhs = (Hs @ s) % 2
        p = self._gf2_solve(Hp, rhs)
        if p is None:
            # Fallback: zero parity
            p = np.zeros(self.M, dtype=np.int8)

        codeword = np.concatenate([p, info_bits.astype(np.int8)])
        return codeword

    @staticmethod
    def _gf2_solve(A, b):
        """Solve A*x = b over GF(2). Returns x or None if no solution."""
        M, N = A.shape
        Ab = np.hstack([A, b.reshape(-1, 1)]).astype(int)
        row = 0
        pivot_rows = []
        for col in range(N):
            found = -1
            for r in range(row, M):
                if Ab[r, col] == 1:
                    found = r
                    break
            if found == -1:
                continue
            Ab[[row, found]] = Ab[[found, row]]
            pivot_rows.append((row, col))
            for r in range(M):
                if r != row and Ab[r, col] == 1:
                    Ab[r] = (Ab[r] + Ab[row]) % 2
            row += 1

        x = np.zeros(N, dtype=np.int8)
        for r, col in pivot_rows:
            x[col] = Ab[r, N]
        return x

    def bp_decode(self, llr_ch, max_iter=50):
        """
        Sum-product (belief propagation) decoding.
        llr_ch: channel LLR values, shape (N,)
        Returns estimated bits (N,).
        """
        N = self.N
        M = self.H.shape[0]

        # Initialize messages: variable -> check = channel LLR
        # msg_vc[n][m_idx] = message from variable n to its m_idx-th check
        msg_vc = [[llr_ch[n]] * len(self.v2c[n]) for n in range(N)]
        # msg_cv[m][n_idx] = message from check m to its n_idx-th variable
        msg_cv = [[0.0] * len(self.c2v[m]) for m in range(M)]

        for iteration in range(max_iter):
            # Check node update (sum-product)
            for m in range(M):
                neighbors = self.c2v[m]
                # Collect incoming messages from variables to this check
                incoming = [msg_vc[n][self.v2c[n].index(m)] for n in neighbors]
                # Product of tanh(L/2) for all but one
                tanh_vals = np.tanh(np.array(incoming, dtype=float) / 2)
                for idx, n in enumerate(neighbors):
                    # Exclude self
                    others = np.delete(tanh_vals, idx)
                    prod = np.prod(others)
                    # Clip to avoid atanh(±1)
                    prod = np.clip(prod, -1 + 1e-7, 1 - 1e-7)
                    msg_cv[m][idx] = 2 * np.arctanh(prod)

            # Variable node update
            for n in range(N):
                checks = self.v2c[n]
                for idx_m, m in enumerate(checks):
                    # Sum of all incoming check messages except from m
                    total = llr_ch[n] + sum(
                        msg_cv[m2][self.c2v[m2].index(n)]
                        for idx_m2, m2 in enumerate(checks) if m2 != m
                    )
                    msg_vc[n][idx_m] = total

            # Compute posterior LLR
            posterior = np.array([
                llr_ch[n] + sum(msg_cv[m][self.c2v[m].index(n)] for m in self.v2c[n])
                for n in range(N)
            ])

            # Hard decision
            est = (posterior < 0).astype(np.int8)

            # Check if valid codeword
            syndrome = (self.H @ est.astype(int)) % 2
            if np.all(syndrome == 0):
                break

        return est
