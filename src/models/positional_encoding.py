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

    def __init__(self, 
                 n_rings: int,
                 n_phi: int, 
                 n_z: int, 
                 d_latent: int
                 ):
        super().__init__()

        assert d_latent % 2 == 0

        self.n_rings = n_rings
        self.n_phi   = n_phi
        self.n_z = n_z
        self.d_latent = d_latent
        

        # -----------------------------
        # Learned embeddings
        # -----------------------------
        self.r_emb = nn.Embedding(self.n_rings, d_latent)
        self.z_emb = nn.Embedding(self.n_z, d_latent)
        self.phi_emb = nn.Embedding(self.n_phi, d_latent)
        # ring-specific modulation (important for irregular geometry)
        self.r_phi_scale = nn.Embedding(self.n_rings, d_latent)

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
        # clamp all indices for safety
        r_idx   = r_idx.long().clamp(0, self.n_rings - 1)
        phi_idx = phi_idx.long().clamp(0, self.n_phi - 1)
        z_idx   = z_idx.long().clamp(0, self.n_z - 1)


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
            angle   = (phi_idx.float() / self.n_phi) * (2.0 * math.pi)
            circ    = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
            phi_emb = phi_emb + self.cont_proj(circ)

        # -----------------------------
        # 4. final sum
        # -----------------------------
        return (
            self.r_emb(r_idx)
            + self.z_emb(z_idx)
            + phi_emb
        )


