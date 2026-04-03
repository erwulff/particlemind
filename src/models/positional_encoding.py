"""
Positional encoding for a cylindrical detector with irregular phi geometry.

Patch coordinates: (r_idx, phi_idx, z_idx)
  - phi_max_per_r[r_idx] gives the number of phi cells for that radial layer.

Each axis is encoded independently and the results are summed:
  - r, z  → learned embeddings
  - phi   → sinusoidal on normalised angle in [0, 2pi), respecting circular topology
"""

import math
import torch
import torch.nn as nn


class DetectorPosEnc(nn.Module):
    def __init__(self, phi_max_per_r: dict[int, int], n_z: int, d_latent: int):
        """
        phi_max_per_r : {r_idx: n_phi_cells}  e.g. {0: 8, 1: 16, 2: 24}
        n_z           : number of z slices
        d_latent      : output embedding dimension (must be even)
        """
        super().__init__()
        assert d_latent % 2 == 0

        n_r = max(phi_max_per_r) + 1
        phi_counts = [phi_max_per_r[r] for r in range(n_r)]
        self.register_buffer("phi_counts", torch.tensor(phi_counts, dtype=torch.float32))

        self.r_emb    = nn.Embedding(n_r, d_latent)
        self.z_emb    = nn.Embedding(n_z, d_latent)
        self.phi_proj = nn.Linear(d_latent, d_latent)

        # log-spaced frequencies for phi sinusoidal encoding
        n_freqs = d_latent // 2
        self.register_buffer("phi_freqs", 2.0 ** torch.arange(n_freqs).float())

    def forward(self, r_idx: torch.Tensor, phi_idx: torch.Tensor, z_idx: torch.Tensor) -> torch.Tensor:
        """
        r_idx, phi_idx, z_idx : LongTensors of shape [...]
        returns                : FloatTensor of shape [..., d_latent]
        """
        # normalise phi to angle in [0, 2pi) using the per-r cell count
        angle = (phi_idx.float() / self.phi_counts[r_idx]) * 2.0 * math.pi
        freqs = angle.unsqueeze(-1) * self.phi_freqs             # [..., n_freqs]
        phi_enc = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # [..., d_latent]

        return self.r_emb(r_idx) + self.phi_proj(phi_enc) + self.z_emb(z_idx)

