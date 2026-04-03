import math
import torch
import torch.nn as nn


class DetectorPosEnc(nn.Module):
    """
    Stable positional encoding for cylindrical detector with irregular phi segmentation.

    Uses:
      - learned r embedding
      - learned z embedding
      - learned phi embedding conditioned on r (IMPORTANT)
      - optional smooth circular features (sin/cos of normalized phi)
    """

    def __init__(self, phi_max_per_r: dict[int, int], n_z: int, d_latent: int):
        super().__init__()

        assert d_latent % 2 == 0

        self.d_latent = d_latent
        self.n_r = max(phi_max_per_r) + 1
        self.n_z = n_z

        # store phi counts per ring
        phi_counts = [phi_max_per_r[r] for r in range(self.n_r)]
        self.register_buffer("phi_counts", torch.tensor(phi_counts, dtype=torch.long))

        # max phi bins across detector (for embedding table sizing)
        self.max_phi = max(phi_counts)

        # -----------------------------
        # Learned embeddings
        # -----------------------------
        self.r_emb = nn.Embedding(self.n_r, d_latent)
        self.z_emb = nn.Embedding(n_z, d_latent)

        # φ embedding (discrete)
        self.phi_emb = nn.Embedding(self.max_phi, d_latent)

        # ring-specific modulation (important for irregular geometry)
        self.r_phi_scale = nn.Embedding(self.n_r, d_latent)

        # -----------------------------
        # Optional continuous circular features
        # -----------------------------
        self.use_continuous = True
        self.cont_proj = nn.Linear(2, d_latent)

    def forward(self, r_idx: torch.Tensor, phi_idx: torch.Tensor, z_idx: torch.Tensor):
        """
        r_idx, phi_idx, z_idx: LongTensors of shape [...]

        Returns:
            [..., d_latent]
        """

        # -----------------------------
        # 1. safety: clamp indices
        # -----------------------------
        r_idx = r_idx.long().clamp(0, self.n_r - 1)
        z_idx = z_idx.long().clamp(0, self.n_z - 1)

        phi_max = self.phi_counts[r_idx]  # [...]

        # prevent invalid phi indices
        phi_idx = phi_idx.long()

        phi_max = self.phi_counts[r_idx]
        
        phi_idx = torch.where(
            phi_idx >= phi_max,
            phi_max - 1,
            phi_idx
        )
        
        phi_idx = torch.clamp(phi_idx, min=0)

        # -----------------------------
        # 2. learned discrete phi embedding
        # -----------------------------
        phi_emb = self.phi_emb(phi_idx)  # [..., D]

        # ring-specific scaling (important for irregular segmentation)
        scale = self.r_phi_scale(r_idx)
        phi_emb = phi_emb * scale

        # -----------------------------
        # 3. optional continuous circular encoding
        # -----------------------------
        if self.use_continuous:
            # normalize within ring safely
            denom = phi_max.clamp(min=1).float()
            angle = (phi_idx.float() / denom) * (2.0 * math.pi)

            circ = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
            circ_emb = self.cont_proj(circ)

            phi_emb = phi_emb + circ_emb

        # -----------------------------
        # 4. final sum
        # -----------------------------
        return (
            self.r_emb(r_idx)
            + self.z_emb(z_idx)
            + phi_emb
        )