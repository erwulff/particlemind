import numpy as np
import torch

def standardize_calo_hit_features_xyz(calo_hit_features):

    calo_hit_features = calo_hit_features.clone()
    
    calo_hit_features[..., 0] /= 1e4
    calo_hit_features[..., 1] /= 1e4
    calo_hit_features[..., 2] /= 1e4
    calo_hit_features[..., 3] = torch.log(calo_hit_features[..., 3] * 1e2) / 10
    return torch.nan_to_num(calo_hit_features, posinf=0.0, neginf=0.0)


def inverse_standardize_calo_hit_features_xyz(calo_hit_features):

    calo_hit_features = calo_hit_features.clone()
    
    calo_hit_features[..., 0] *= 1e4
    calo_hit_features[..., 1] *= 1e4
    calo_hit_features[..., 2] *= 1e4
    calo_hit_features[..., 3] = torch.exp(calo_hit_features[..., 3] * 10) / 1e2
    return calo_hit_features



def add_random_noise(x, std=1e-2):
    noise = torch.randn_like(x) * std
    noise[..., 3] = torch.abs(noise[..., 3])
    return x + noise


def global_phi_rotation(x):
    """
    Assumes feature order: (x, y, z, E) or similar,
    """


    phi_angle = torch.rand(1, device=x.device) * 2.0 * torch.pi

    # apply rotation
    x = x.clone()
    x_cos = torch.cos(phi_angle)
    x_sin = torch.sin(phi_angle)
    x_new = x[..., 0] * x_cos - x[..., 1] * x_sin
    y_new = x[..., 0] * x_sin + x[..., 1] * x_cos
    x[..., 0] = x_new
    x[..., 1] = y_new
    

    return x

  
def augment_data(x):
    """
    Apply stochastic augmentations.
    """

    x = x.clone()  # 🔴 critical: global safety clone

    # 1. random noise
    x = add_random_noise(x, std=1e-2)

    ## 2. global phi rotation
    x = global_phi_rotation(x)

    return x
