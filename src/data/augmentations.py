import numpy as np
import torch



def standardize_calo_hit_features_xyz(calo_hit_features):

    calo_hit_features = np.copy(calo_hit_features)
    
    calo_hit_features[..., 0] /= 1e4
    calo_hit_features[..., 1] /= 1e4
    calo_hit_features[..., 2] /= 1e4
    calo_hit_features[..., 3] = np.log(calo_hit_features[..., 3] * 1e2) / 10
    return np.nan_to_num(calo_hit_features, posinf=0.0, neginf=0.0)


def inverse_standardize_calo_hit_features_xyz(calo_hit_features):

    calo_hit_features = np.copy(calo_hit_features)
    
    calo_hit_features[..., 0] *= 1e4
    calo_hit_features[..., 1] *= 1e4
    calo_hit_features[..., 2] *= 1e4
    calo_hit_features[..., 3] = np.exp(calo_hit_features[..., 3] * 10) / 1e2
    return calo_hit_features

def add_random_noise(x, frac=0.05):
    noise = np.random.normal(size=x.shape) * (np.abs(x) * frac)
    noise[..., 3] = np.abs(noise[..., 3])
    return x + noise


def global_phi_rotation(x, max_angle=2.0*np.pi):
    """
    Assumes feature order: (x, y, z, E) or similar,
    """

    phi_angle = np.random.random(1) * max_angle
    phi_angle = np.pi

    # apply rotation
    x = np.copy(x)
    x_cos = np.cos(phi_angle)
    x_sin = np.sin(phi_angle)
    x_new = x[..., 0] * x_cos - x[..., 1] * x_sin
    y_new = x[..., 0] * x_sin + x[..., 1] * x_cos
    x[..., 0] = x_new
    x[..., 1] = y_new
    

    return x


def collinear_split(x, frac=0.01):
    """
    Randomly split frac of the hits into two collinear hits with a random amount of energy
    """
    x = np.copy(x)

    num_hits = x.shape[0]
    num_split = int(num_hits * frac)

    if num_split == 0:
        return x

    split_indices = np.random.choice(num_hits, size=num_split, replace=False)

    new_hits = []
    for idx in split_indices:
        hit = x[idx]
        energy = hit[3]
        if energy <= 0:
            continue
        split_energy = energy * np.random.uniform(0.1, 0.9)
        hit[3] -= split_energy
        new_hit = np.copy(hit)
        new_hit[3] = split_energy
        new_hits.append(new_hit)

    if new_hits:
        x = np.concatenate([x, np.array(new_hits)], axis=0)

    return x




  
def augment_data(x):
    """
    Apply stochastic augmentations.
    """

    x = np.copy(x)  # 🔴 critical: global safety clone

    # 1. random noise
    x = add_random_noise(x)

    ## 2. global phi rotation
    x = global_phi_rotation(x, max_angle = 0.5*np.pi)

    ## 3. collinear split
    x = collinear_split(x)

    return x
