import numpy as np
import torch

def standardize_calo_hit_features_xyz(calo_hit_features):
    calo_hit_features[..., 0] /= 1e4
    calo_hit_features[..., 1] /= 1e4
    calo_hit_features[..., 2] /= 1e4
    calo_hit_features[..., 3] = np.log(calo_hit_features[..., 3] * 1e2) / 10
    return calo_hit_features


def inverse_standardize_calo_hit_features_xyz(calo_hit_features):
    calo_hit_features[..., 0] *= 1e4
    calo_hit_features[..., 1] *= 1e4
    calo_hit_features[..., 2] *= 1e4
    calo_hit_features[..., 3] = np.exp(calo_hit_features[..., 3] * 10) / 1e2
    return calo_hit_features


def standardize_calo_hit_features_rphiz(calo_hit_features):
    calo_hit_features[..., 0] /= 1e4
    calo_hit_features[..., 1] /= 1e1
    calo_hit_features[..., 2] /= 1e4
    calo_hit_features[..., 3] = torch.log(calo_hit_features[..., 3] * 1e2) / 10
    return calo_hit_features


def inverse_standardize_calo_hit_features_rphiz(calo_hit_features):
    calo_hit_features[..., 0] *= 1e4
    calo_hit_features[..., 1] *= 1e1
    calo_hit_features[..., 2] *= 1e4
    calo_hit_features[..., 3] = np.exp(calo_hit_features[..., 3] * 10) / 1e2
    return calo_hit_features



def add_random_noise(x, std=1e-2):
    noise = torch.randn_like(x) * std
    noise[..., 3] = torch.abs(noise[..., 3])
    return x + noise


def global_phi_rotation(x):
    """
    Assumes feature order: (r, phi, theta, E) or similar,
    where phi is index 1.
    """

    phi_angle = torch.rand(1, device=x.device) * 2.0 * torch.pi

    # apply rotation
    x = x.clone()
    x[..., 1] = x[..., 1] + phi_angle

    # wrap phi to [-pi, pi]
    x[..., 1] = (x[..., 1] + torch.pi) % (2 * torch.pi) - torch.pi

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
