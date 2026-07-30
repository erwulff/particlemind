import torch
import torch.nn as nn
from src.models.positional_encoding import DetectorPosEnc
import torch.nn.functional as F

# vqtorch can be installed from https://github.com/minyoungg/vqtorch
try:
    from vqtorch.nn import VectorQuant  # type: ignore
except ImportError as e:
    raise ImportError("vqtorch is not installed. Please install it to use this module.") from e


def safe(x):
    if torch.is_tensor(x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return x
    if isinstance(x, float):
        if not np.isfinite(x):
            return 0.0
    return x



class VQVAEMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=2,
        latent_dim=2,
        data_type="",
        encoder_layers=None,
        decoder_layers=None,
        vq_kwargs=None,
        **kwargs,
    ):
        """Initializes the VQ-VAE model.

        Parameters
        ----------
        codebook_size : int, optional
            The size of the codebook. The default is 8.
        embed_dim : int, optional
            The dimension of the embedding space. The default is 2.
        input_dim : int, optional
            The dimension of the input data. The default is 2.
        encoder_layers : list, optional
            List of integers representing the number of units in each encoder layer.
            If None, a default encoder with a single linear layer is used. The default is None.
        decoder_layers : list, optional
            List of integers representing the number of units in each decoder layer.
            If None, a default decoder with a single linear layer is used. The default is None.
        """

        super().__init__()

        self.loss_history = []
        self.lr_history = []

        self.vq_kwargs = vq_kwargs
        self.embed_dim = latent_dim
        self.input_dim = input_dim  # for jet constituents, eta and phi
        self.data_type = data_type

        # --- Encoder --- #
        if encoder_layers is None:
            self.encoder = torch.nn.Linear(self.input_dim, self.embed_dim)
        else:
            enc_layers = []
            enc_layers.append(torch.nn.Linear(self.input_dim, encoder_layers[0]))
            enc_layers.append(torch.nn.ReLU())

            for i in range(len(encoder_layers) - 1):
                enc_layers.append(torch.nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
                enc_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.Linear(encoder_layers[-1], self.embed_dim))

            self.encoder = torch.nn.Sequential(*enc_layers)

        

        # --- Decoder --- #
        if decoder_layers is None:
            self.decoder = torch.nn.Linear(self.embed_dim, self.input_dim)
        else:
            dec_layers = []
            dec_layers.append(torch.nn.Linear(self.embed_dim, decoder_layers[0]))
            dec_layers.append(torch.nn.ReLU())

            for i in range(len(decoder_layers) - 1):
                dec_layers.append(torch.nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
                dec_layers.append(torch.nn.ReLU())
            dec_layers.append(torch.nn.Linear(decoder_layers[-1], self.input_dim))

            self.decoder = torch.nn.Sequential(*dec_layers)


        # --- Vector-quantization layer --- #
        if self.vq_kwargs is not None:
            self.vqlayer = VectorQuant(feature_size=self.embed_dim, **vq_kwargs)

    def forward(self, batch, x, mask):

        if self.data_type == "hit":

            """
            Inputs:
                batch: TENSOR
            """

            
            # mask is there for compatibility with the transformer model
            # encode
            z_embed = self.encoder(x)
            # quantize
            if self.vq_kwargs is not None:
                z_q2, vq_out = self.vqlayer(z_embed)
            else:
                z_q2, vq_out = z_embed, None
            # decode
            x_reco = self.decoder(z_q2)
            
            return x_reco, vq_out, z_embed
    
    





class NormformerBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # define the MultiheadAttention layer with layer normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(input_dim)

        # define the MLP with layer normalization
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),  # Add layer normalization
            nn.Linear(input_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(mlp_dim, input_dim),
        )

        # initialize weights of mlp[-1] and layer norm after attn block to 0
        # such that the residual connection is the identity when the block is
        # initialized
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.norm1.weight)

    def forward(self, x, mask=None, return_attn_weights=False):
        # x: (B, S, F)
        # mask: (B, S)
        x = x * mask.unsqueeze(-1)

        # calculate self-attention
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask != 1)
        # Add residual connection and permute back to (B, S, F)
        attn_res = self.norm2(attn_output) + x

        output = self.mlp(attn_res) + attn_res

        if return_attn_weights:
            return output, attn_weights

        # output shape: (B, S, F)
        return output


class Transformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        skip_out_proj=False,
    ):
        super().__init__()

        self.project_in = nn.Linear(input_dim, hidden_dim)

        self.num_blocks = num_blocks
        self.skip_out_proj = skip_out_proj
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.blocks = nn.ModuleList(
            [NormformerBlock(input_dim=hidden_dim, mlp_dim=hidden_dim, num_heads=num_heads) for _ in range(num_blocks)]
        )
        self.project_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask):
        x = self.project_in(x)
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask)
        if self.skip_out_proj:
            return x * mask.unsqueeze(-1)
        x = self.project_out(x) * mask.unsqueeze(-1)
        return x


class NormformerStack(torch.nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=1,
        num_blocks=2,
        skip_out_proj=False,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.skip_out_proj = skip_out_proj
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.blocks = nn.ModuleList(
            [
                NormformerBlock(
                    input_dim=self.hidden_dim,
                    mlp_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, mask):
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask)
        return x * mask.unsqueeze(-1)






class VQVAENormFormer(torch.nn.Module):
    """This is basically just a re-factor of the VQVAETransformer class, but with more modular
    model components, making it easier to use some components in other models."""

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        data_type="",
        num_heads=1,
        num_blocks=2,
        vq_kwargs=None,
        vit_kwargs=None,
        **kwargs,
    ):
        super().__init__()

        self.loss_history = []
        self.lr_history = []

        self.vq_kwargs = vq_kwargs
        self.vit_kwargs = vit_kwargs
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.data_type = data_type
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # Model components:
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder_normformer = NormformerStack(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
        )
        self.latent_projection_in = nn.Linear(self.hidden_dim, self.latent_dim)
        self.latent_projection_out = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_normformer = NormformerStack(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
        )
        self.output_projection = nn.Linear(hidden_dim, input_dim)
    
        if self.vq_kwargs is not None:
          self.vqlayer = VectorQuant(feature_size=self.latent_dim, **vq_kwargs)
          
          
        if self.vit_kwargs is not None:

          # ViT components
          # each patch size needs its own linear encoder
          self.linear_projection_encoders = torch.nn.ModuleDict()
          for key in vit_kwargs["unique_patch_sizes_dict"].keys():
              self.linear_projection_encoders[str(key)] = torch.nn.Linear(vit_kwargs["unique_patch_sizes_dict"][key], vit_kwargs["D_EMBEDDING"], bias=False) # no bias so transpose = inverse

          self.positional_encoding = DetectorPosEnc(
              n_rings = vit_kwargs["n_rings"],
              n_phi = vit_kwargs["n_phi_patches"],
              n_z =  vit_kwargs["n_bins_z"],
              d_latent = vit_kwargs["D_EMBEDDING"],
          )        

    
    def forward(self, batch, x, mask):
        
        if self.data_type == "patch":

            """
            Inputs:
                batch: dict with keys = tuples associated with the unique patch index (# r cells, # phi cells, # z cells)
                batch[key]: dict with keys:
                    flat_tensor
                    global_patch_ids
                    mask
    
            Returns:
                e
                e_reco
                {key:batch[key]["flat_tensor"] for key in batch.keys()}, 
                x_reco_chunks
                vq_out
            """

            # print out the number of zero patches per event
            # for key in batch.keys():
            #     mask = batch[key]['mask']  # (B, P_k)
            #     zero_per_event = (mask == 1).sum(dim=1)  # (B,)
            #     total_patches = mask.shape[1]
            #     print(f"Key: {key}")
            #     print(f"  nonzero patches per event: {zero_per_event.tolist()}")
            #     print(f"  total patches per event: {total_patches}")
            #     print()

    
            embeddings = []
            global_patch_ids = []
            local_patch_ids = []
            patch_mask = []
            patch_group_sizes = []  # track how many patches per group, for splitting later

            keys = list(sorted(batch.keys()))
        
            # 1. encode each patch group
            # endcap_neg shares projection weights with endcap_pos (mirror symmetry).
            for key in keys:
                encoder_key = key.replace("endcap_neg_", "endcap_pos_")
                emb = self.linear_projection_encoders[str(encoder_key)](batch[key]["flat_tensor"])  # (B, P_k, D)



                P_k = emb.shape[1]
                embeddings.append(emb)
                global_patch_ids.append(batch[key]["global_patch_ids"])
                local_patch_ids.append(batch[key]["local_patch_ids"])
                patch_mask.append(batch[key]["mask"])
                patch_group_sizes.append(P_k)
    
            # 2. concatenate all patches
            embeddings       = torch.cat(embeddings,       dim=1)  # (B, P_total, D)
            global_patch_ids = torch.cat(global_patch_ids, dim=1)  # (B, P_total)
            local_patch_ids  = torch.cat(local_patch_ids,  dim=1)  # (B, P_total, 3)
            patch_mask       = torch.cat(patch_mask,             dim=1)  # (B, P_total)


            # 3. reorder all tensors by global patch id
            order   = torch.argsort(global_patch_ids, dim=1)
            order_D = order.unsqueeze(-1).expand_as(embeddings)
            order_3 = order.unsqueeze(-1).expand_as(local_patch_ids)
        
            embeddings      = torch.gather(embeddings,      dim=1, index=order_D)
            local_patch_ids = torch.gather(local_patch_ids, dim=1, index=order_3)
            patch_mask      = torch.gather(patch_mask,            dim=1, index=order)
        
            # 4. add positional encoding
            r_idx, phi_idx, z_idx = local_patch_ids[..., 0], local_patch_ids[..., 1], local_patch_ids[..., 2]
    
            e = embeddings + self.positional_encoding(r_idx, phi_idx, z_idx)  # (B, P_total, D)
    
        
            # 5. encode → quantize → decode
            e       = self.input_projection(e)
            e       = self.encoder_normformer(e, mask=patch_mask)
            z_embed = self.latent_projection_in(e) * patch_mask.unsqueeze(-1)
    
            if self.vq_kwargs is not None:
                B_vq, P_vq, D_vq = z_embed.shape
                flat_mask = patch_mask.reshape(B_vq * P_vq).bool()   # (B*P,)
                z_flat    = z_embed.reshape(B_vq * P_vq, D_vq)       # (B*P, D)

                # Run VQ only on non-zero-energy patches so masked positions
                # don't contaminate codebook assignments or the commitment loss.
                valid_z_q, vq_out = self.vqlayer(z_flat[flat_mask].unsqueeze(0))  # (1, K, D)
                valid_z_q = valid_z_q.squeeze(0)                                   # (K, D)

                # Scatter quantized results back to full (B, P, D) shape.
                z_flat_q = torch.zeros_like(z_flat)
                z_flat_q[flat_mask] = valid_z_q
                z = z_flat_q.reshape(B_vq, P_vq, D_vq)

                # Rebuild vq_out tensors from (1, K, ...) → (B, P, ...) so that
                # downstream code indexing [b][rank] keeps working.
                for out_key in ["q", "z", "z_q"]:
                    if out_key not in vq_out or not torch.is_tensor(vq_out[out_key]):
                        continue
                    val = vq_out[out_key].squeeze(0)          # (K, *trailing)
                    trailing = val.shape[1:]
                    full = torch.zeros(B_vq * P_vq, *trailing, dtype=val.dtype, device=val.device)
                    full[flat_mask] = val
                    vq_out[out_key] = full.reshape(B_vq, P_vq, *trailing)
               
            else:
                z, vq_out = z_embed, None

            e_reco  = self.latent_projection_out(z) * patch_mask.unsqueeze(-1)
            e_reco  = self.decoder_normformer(e_reco, mask=patch_mask)
            e_reco  = self.output_projection(e_reco) * patch_mask.unsqueeze(-1)
        
            # 6. undo the sort so patches line up with their original key groupings
            # argsort of argsort gives the inverse permutation
            inv_order = torch.argsort(order, dim=1)
            e_reco_unordered = torch.gather(e_reco, dim=1, index=inv_order.unsqueeze(-1).expand_as(e_reco))
        
            # 7. split back by patch group and decode each with its own linear decoder
            x_reco_chunks = {}
            start = 0
            for key, P_k in zip(keys, patch_group_sizes):
                chunk = e_reco_unordered[:, start:start + P_k, :]      # (B, P_k, D)
                encoder_key = key.replace("endcap_neg_", "endcap_pos_")
                W = self.linear_projection_encoders[str(encoder_key)].weight  # (D, bins_k)
                x_reco_chunks[key] = F.linear(chunk, W.T)  # (B, P_k, bins_k)
                start += P_k
    
            return e, e_reco, {key:batch[key]["flat_tensor"] for key in batch.keys()}, x_reco_chunks, vq_out

        elif self.data_type == "hit":


            """
            Inputs:
                batch: TENSOR
            """
            # encode
         #   print("x", torch.sum(torch.isnan(x)))
            x0 = self.input_projection(x) # BS, num hits, hidden_dim
           # print("x0", torch.sum(torch.isnan(x0)))
            x1 = self.encoder_normformer(x0, mask=mask) # BS, num hits, hidden_dim
           # print("x1", torch.sum(torch.isnan(x1)))
            z_embed = self.latent_projection_in(x1) * mask.unsqueeze(-1)  # BS, num hits, latent_dim
           # print("z_embed", torch.sum(torch.isnan(z_embed)))
            
            # quantize
            if self.vq_kwargs is not None:
                z, vq_out = self.vqlayer(z_embed) # BS, num hits, latent_dim
            else:
                z, vq_out = z_embed, None

           # print("z", torch.sum(torch.isnan(z)))

            
            # decode
            x_reco0 = self.latent_projection_out(z) * mask.unsqueeze(-1) # BS, num hits, hidden_dim
           # print("x_reco0", torch.sum(torch.isnan(x_reco0)))
            x_reco1 = self.decoder_normformer(x_reco0, mask=mask) # BS, num hits, hidden_dim
           # print("x_reco1", torch.sum(torch.isnan(x_reco1)))
            x_reco = self.output_projection(x_reco1) * mask.unsqueeze(-1) # BS, num hits, input_dim
          #  print("x_reco", torch.sum(torch.isnan(x_reco)))

          
            
            return x_reco, vq_out, z_embed # CHANGED


def mse_loss(x_hit_truth, x_hit_reco, mask):
    diff = (x_hit_reco - x_hit_truth) ** 2      # (B, H, 4)
    diff = diff.sum(dim=-1)                     # (B, H)  sum over features
    reco_loss = (diff * mask).sum() / mask.sum().clamp(min=1)
    return reco_loss


def chamfer_distance_torch(A, B, mask=None):
    """
    Chamfer distance using PyTorch — runs on GPU if available.

    Args:
        A: torch.Tensor of shape (N, D) or (B, N, D) for batched input
        B: torch.Tensor of shape (N, D) or (B, N, D) for batched input
        mask: bool torch.Tensor of shape (B, N) — which hits are real.
              If None, all hits are treated as real (unbatched use case).

    Returns:
        Scalar tensor (mean chamfer distance across batch)
    """

    batched = A.dim() == 3
    if not batched:
        # Unbatched: just compute directly, no mask needed
        diff = A.unsqueeze(1) - B.unsqueeze(0)       # (N, M, D)
        sq_dist = (diff ** 2).sum(dim=-1)            # (N, M)
        a_to_b = sq_dist.min(dim=1).values.sqrt().mean()
        b_to_a = sq_dist.min(dim=0).values.sqrt().mean()
        return a_to_b + b_to_a

    # Batched: must process per-item to avoid cross-event comparisons
    assert mask is not None, "mask is required for batched Chamfer distance"
    assert mask.shape == A.shape[:2], "mask shape must be (B, N)"

    batch_size = A.shape[0]
    chamfer_vals = []

    for i in range(batch_size):
        m = mask[i]                  # (N,) bool
        a = A[i][m]                  # (K, D) — real hits only
        b = B[i][m]                  # (K, D) — real hits only

        if a.shape[0] == 0:
            # No real hits in this event; contribute zero loss
            chamfer_vals.append(A.new_zeros(1).squeeze())
            continue

        diff = a.unsqueeze(1) - b.unsqueeze(0)      # (K, K, D)
        sq_dist = (diff ** 2).sum(dim=-1)           # (K, K)

        a_to_b = sq_dist.min(dim=1).values.sqrt().mean()
        b_to_a = sq_dist.min(dim=0).values.sqrt().mean()
        chamfer_vals.append(a_to_b + b_to_a)

    return torch.stack(chamfer_vals).mean()



    

def reco_loss_function(x_hit_truth, x_hit_reco, mask, loss_type="mse"):
    mask = mask.bool()
    if loss_type == "mse":
        loss = mse_loss(x_hit_truth, x_hit_reco, mask)
    elif loss_type == "chamfer":
        loss = chamfer_distance_torch(x_hit_truth, x_hit_reco, mask=mask)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return loss


def mean_knn_distance(x, mask, k=5):
    """
    x : (B,N,4)
    mask : (B,N)
    """

    out = []

    for pts, m in zip(x, mask):
        pts = pts[m.bool()]

        if pts.shape[0] <= k:
            out.append(torch.tensor(0., device=x.device))
            continue

        dist = torch.cdist(pts, pts)

        # ignore self
        dist.fill_diagonal_(float("inf"))

        knn = dist.topk(k, largest=False).values

        out.append(knn.mean())

    return torch.stack(out).mean()