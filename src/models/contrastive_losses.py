import torch
import torch.nn.functional as F


def pool(z, mask):
    # z: (B, N, 1, D)
    z = z.squeeze(2) # (B, N, D)
    mask = mask.unsqueeze(-1)
    return (z * mask).sum(dim=1) / mask.sum(dim=1)


def SimCLR_loss(z1, z2, mask1, mask2, temperature=0.1, alpha=1):


    z1 = pool(z1, mask1)
    z2 = pool(z2, mask2)
    # inputs have shape (B, latent_dim)


    batch_size = z1.shape[0]
    z1 = F.normalize( z1, dim=1 )
    z2 = F.normalize( z2, dim=1 )
    z   = torch.cat( [z1, z2], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size ) # batch_size above the main diagonal -- x_i x_i'
    sim_ji = torch.diag( similarity_matrix, -batch_size ) # below the main diagonal -- x_i' x_i
    positives = torch.cat( [sim_ij, sim_ji], dim=0 )
    nominator = torch.exp( positives / temperature )
    negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float().to(z1.device)
    denominator = negatives_mask * torch.exp( similarity_matrix / temperature )
    loss_partial = -torch.log( nominator / (torch.sum( denominator, dim=1 )).pow(exponent=alpha) )
    loss = torch.sum( loss_partial )/( 2*batch_size )

    return loss


def CLIP_loss(z1, z2, mask1, mask2, temperature=0.07):
    z1 = pool(z1, mask1)
    z2 = pool(z2, mask2)
    # inputs have shape (B, latent_dim)
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Compute similarity matrix: (B, B)
    similarity_matrix = torch.matmul(z1, z2.T) / temperature

    # Labels: diagonal entries are positives
    labels = torch.arange(batch_size, device=z1.device)

    # Symmetric cross-entropy loss
    loss_i = F.cross_entropy(similarity_matrix, labels)         # z1 -> z2
    loss_j = F.cross_entropy(similarity_matrix.T, labels)       # z2 -> z1

    return (loss_i + loss_j) / 2