import numpy as np

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

"""
detector_vit_patching.py

ViT-style patching for LHC-like barrel detector geometry (ECAL/HCAL barrel).

Pipeline:
  1. build_patch_registry(...)   → describes every patch: its (r_idx, phi_idx, z_idx)
                                   and its continuous (r, phi, z) centre coordinates.
  2. assign_hits_to_patches(...) → maps (x,y,z) hits onto cell indices, then onto a
                                   patch token.  Returns a dict keyed by patch_id with
                                   the flat cell tensor + positional encoding.

Coordinate conventions
----------------------
  r    – radial ring index (same as layer group index)
  phi  – azimuthal patch index (0 … N_phi-1, wrapping)
  z    – longitudinal patch index (0 … N_z-1)

Position encoding
-----------------
  Learnable sinusoidal encoding in 3D.  Each patch gets a D-dimensional vector
  built by concatenating three independent sinusoidal encodings (one per axis),
  each of dimension D//3.  If D is not divisible by 3 the remainder goes to z.
"""


def build_patch_registry(
    detector_patching_params: Dict,
    num_wedges: int = 16,
):
    """
    Build the complete patch registry for a barrel sub-detector.

    Parameters mirror those passed to ``do_detector_patching`` exactly so
    that the patch boundaries are consistent with the visualisation code.

    Returns
    -------
    PatchRegistry
    """

    cells_per_wedge = detector_patching_params["barrel_configs"]["cells_per_wedge"]
    cells_per_patch_phi = detector_patching_params["barrel_configs"]["cells_per_patch_phi"]
    cells_per_patch_z = detector_patching_params["barrel_configs"]["cells_per_patch_z"]
    cell_size = detector_patching_params["cell_size"]
    layer_width = detector_patching_params["layer_width"]
    num_cells_z = detector_patching_params["barrel_configs"]["n_bins_z"]
    r_start = detector_patching_params["barrel_configs"]["x_start_midpoint"] - layer_width / 2
    r_stop = detector_patching_params["barrel_configs"]["x_stop_midpoint"] + layer_width / 2
    z_start = detector_patching_params["barrel_configs"]["z_start_midpoint"] - cell_size / 2
    z_stop = detector_patching_params["barrel_configs"]["z_stop_midpoint"] + cell_size / 2
    offsets = detector_patching_params["barrel_configs"]["offsets"]
    groups = detector_patching_params["barrel_configs"]["groups"]

    
    registry = {
        "patches": [],                  # list of dicts (was PatchInfo objects)
        "index": {},                   # (r,phi,z) -> patch_id
        "n_phi_per_ring": {},
        "n_z_patches": 0,
        "n_rings": 0,
    }

    
    n_layers = len(cells_per_wedge)
    n_z_patches = num_cells_z // cells_per_patch_z
    assert num_cells_z % cells_per_patch_z == 0, \
        "num_cells_z must be divisible by cells_per_patch_z"

    # --- radial (r) geometry: one ring per layer ---
    r_edges = np.linspace(r_start, r_stop, n_layers + 1)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2

    # --- z geometry (same for all rings) ---
    z_edges = np.linspace(z_start, z_stop, num_cells_z + 1)
    z_cell_centers = (z_edges[:-1] + z_edges[1:]) / 2
    z_patch_centers = np.array([
        np.mean(z_cell_centers[j*cells_per_patch_z:(j+1)*cells_per_patch_z])
        for j in range(n_z_patches)
    ])

    # --- map each layer index to its ring (group) index ---
    #     groups = [0, 3, 6, …]  means layer 0,1,2 → ring 0; 3,4,5 → ring 1 …
    layer_to_ring = {}
    ring_to_layers = {}
    for ring_idx, (g_start, g_stop) in enumerate(zip(groups[:-1], groups[1:])):
        ring_to_layers[ring_idx] = (g_start, g_stop - 1)
        for layer in range(g_start, g_stop):
            layer_to_ring[layer] = ring_idx

    # verify all layers share the same cells_per_wedge within a ring
    for ring_idx, (l_start, l_stop) in ring_to_layers.items():
        cpw = cells_per_wedge[l_start]
        for l in range(l_start, l_stop + 1):
            assert cells_per_wedge[l] == cpw, \
                f"Ring {ring_idx}: layers {l_start}–{l_stop} must share cells_per_wedge"

    # --- build patches ring by ring ---
    patch_id = 0
    registry["n_z_patches"] = n_z_patches
    registry["n_rings"] = len(groups) - 1


    for ring_idx, (l_start, l_stop) in ring_to_layers.items():


        # use the first layer of the ring for geometry
        ref_layer = l_start
        cpw = cells_per_wedge[ref_layer]                    # phi-cells per wedge
        cpp = cells_per_patch_phi[ref_layer]                # phi-cells per patch
        total_cells_ring = cpw * num_wedges                 # total phi-cells in ring
        n_phi_patches = total_cells_ring // cpp
        assert total_cells_ring % cpp == 0, \
            f"Ring {ring_idx}: total_cells_ring={total_cells_ring} not divisible by cpp={cpp}"

        registry["n_phi_per_ring"][ring_idx] = n_phi_patches

        # -- phi patch centres --
        # phi cell indices run 0 … total_cells_ring-1 around the ring.
        # Cell i sits at phi = 2π * (i + 0.5) / total_cells_ring
        # Patch phi_idx spans cells [phi_idx*cpp … (phi_idx+1)*cpp - 1]
        phi_patch_centers = np.array([
            2 * np.pi * ((phi_idx * cpp) + cpp / 2) / total_cells_ring
            for phi_idx in range(n_phi_patches)
        ])

        # -- r centre for this ring --

        r_c = np.mean(r_centers[range(l_start, l_stop + 1)])   # approximate as the first layer's r centre


        for phi_idx in range(n_phi_patches):
            phi_cell_start = phi_idx * cpp
            phi_cell_stop  = phi_cell_start + cpp - 1

            for z_idx in range(n_z_patches):
                z_cell_start = z_idx * cells_per_patch_z
                z_cell_stop  = z_cell_start + cells_per_patch_z - 1

                info = {
                    "patch_id": patch_id,
                    "r_idx": ring_idx,
                    "phi_idx": phi_idx,
                    "z_idx": z_idx,
                    "r_center": float(r_c),
                    "phi_center": float(phi_patch_centers[phi_idx]),
                    "z_center": float(z_patch_centers[z_idx]),
                    "layer_start": l_start,
                    "layer_stop": l_stop,
                    "phi_cell_start": phi_cell_start,  
                    "phi_cell_stop": phi_cell_stop, 
                    "z_cell_start": z_cell_start,      
                    "z_cell_stop": z_cell_stop,         
                }
                
                registry["patches"].append(info)
                registry["index"][(ring_idx, phi_idx, z_idx)] = patch_id
                patch_id += 1

    tmp1 = registry["n_rings"]
    

    print(f"[Registry] {patch_id} total patches")
    print(f"           {tmp1} rings ")
    print(f"           {n_z_patches} z-bins")
    print("           ring: total number of phi patches ")
    for r in sorted(registry["n_phi_per_ring"]):
        tmp2 = registry["n_phi_per_ring"][r]
        
        print(f"              {r}: {tmp2}")

    
    unique_patch_sizes = defaultdict(int)
    # Pre-allocate cell tensors for every patch
    for p in registry["patches"]:
        l_start = p["layer_start"]
        l_stop = p["layer_stop"]

        n_layers_in_ring = l_stop - l_start + 1
        cpp = cells_per_patch_phi[l_start]
        n_z = cells_per_patch_z

        unique_patch_sizes[(n_layers_in_ring, cpp, n_z)] += 1
    

    return registry, unique_patch_sizes, patch_id




def _hits_to_cell_indices_barrel(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    detector_patching_params: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-implementation of assign_hits_to_cells from the notebook, returning
    (sector_idx, phi_global_cell_idx, z_cell_idx) for each hit.

    phi_global_cell_idx is the global phi index across the full ring
    (not the within-wedge index).
    """
    n_sectors = 16
    half_wedge = np.pi / n_sectors

    cell_size   = detector_patching_params["cell_size"]
    layer_width = detector_patching_params["layer_width"]
    
    cfg = detector_patching_params["barrel_configs"]
    n_bins_x = cfg["n_bins_x"]
    n_bins_y = cfg["n_bins_y"]
    n_bins_z = cfg["n_bins_z"]
    x_start = cfg["x_start_midpoint"] - layer_width / 2
    x_stop  = cfg["x_stop_midpoint"]  + layer_width / 2
    y_start = cfg["y_start_midpoint"] - cell_size / 2
    y_stop  = cfg["y_stop_midpoint"]  + cell_size / 2
    z_start = cfg["z_start_midpoint"] - cell_size / 2
    z_stop  = cfg["z_stop_midpoint"]  + cell_size / 2

    cells_per_wedge = cfg["cells_per_wedge"]
    offsets = cfg["offsets"]



    # wedge / sector
    angles = np.arctan2(y, x)
    angles[angles < 0] += 2 * np.pi
    angles_shifted = (angles + half_wedge) % (2 * np.pi)
    sector_idx = (angles_shifted / (2 * np.pi) * n_sectors).astype(int)



    # rotate to sector-0 frame
    delta_theta   = 2 * np.pi / n_sectors
    theta_center  = (sector_idx + 0.5) * delta_theta
    theta_center0 = 0.5 * delta_theta
    theta_rot = -(theta_center - theta_center0)
    x_local = x * np.cos(theta_rot) - y * np.sin(theta_rot)
    y_local = x * np.sin(theta_rot) + y * np.cos(theta_rot)

    x_edges = np.linspace(x_start, x_stop, n_bins_x + 1)
    y_edges = np.linspace(y_start, y_stop, n_bins_y + 1)
    z_edges = np.linspace(z_start, z_stop, n_bins_z + 1)

  

    layer_cell_idx = np.digitize(x_local, x_edges) - 1   # radial layer
    y_cell_idx     = np.digitize(y_local, y_edges) - 1   # phi within wedge
    z_cell_idx     = np.digitize(z, z_edges) - 1

    # global phi index (0 … total_cells_ring-1)

    # adjustment for layers
    max_layer = max(cells_per_wedge.keys()) + 1
    
    cpw_arr = np.array([cells_per_wedge[i] for i in range(max_layer)])
    offset_arr = np.array([offsets[35 - i] for i in range(max_layer)])
    
    y_cell_adjusted = y_cell_idx - offset_arr[layer_cell_idx]
    
    phi_global = (
        sector_idx * cpw_arr[layer_cell_idx]
        + y_cell_adjusted
    )
    
    # check for out-of-range hits
    bad = (
        (layer_cell_idx < 0) | (layer_cell_idx >= n_bins_x) |
        (y_cell_idx < 0)     | (y_cell_idx >= n_bins_y) |
        (z_cell_idx < 0)     | (z_cell_idx >= n_bins_z)
    )
    if bad.any():
        print(f"{bad.sum()} hits out of range and will be ignored")
        exit()

    return layer_cell_idx, phi_global, z_cell_idx, ~bad





def assign_hits_to_patches_barrel(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    energy: np.ndarray,             # per-hit energy deposit (or any scalar feature)
    registry: dict,
    detector_patching_params: dict,
    embed_dim: int = 128,
) -> Dict[int, Dict]:
    """
    Assign hits to ViT patches and return per-patch data.

    Parameters
    ----------
    x, y, z       : hit coordinates [mm]
    energy        : hit energy deposits
    layer_indices : which detector layer each hit belongs to (0-based within barrel)
    registry      : built by build_patch_registry(...)
    groups        : same list passed to build_patch_registry
    cells_per_wedge, cells_per_patch_phi, cells_per_patch_z : geometry params
    detector_patching_params : the yaml dict
    embed_dim       : dimensionality of the positional encoding

    Returns
    -------
    dict keyed by patch_id, each value is:
        {
          "patch_info"    : PatchInfo,
          "cell_tensor"   : np.ndarray shape (n_layers_in_ring, n_phi_cells, n_z_cells),
          "true_coordinates"  : (physical r center, physical phi center, physical z center)
          "n_hits"        : int,
        }
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    energy = np.asarray(energy, dtype=float)

    groups = detector_patching_params["barrel_configs"]["groups"]
    cells_per_patch_phi = detector_patching_params["barrel_configs"]["cells_per_patch_phi"]
    cells_per_patch_z = detector_patching_params["barrel_configs"]["cells_per_patch_z"]

    # layer → ring mapping
    layer_to_ring = {}
    ring_to_layers: Dict[int, Tuple[int,int]] = {}
    for ring_idx, (g_start, g_stop) in enumerate(zip(groups[:-1], groups[1:])):
        ring_to_layers[ring_idx] = (g_start, g_stop - 1)
        for l in range(g_start, g_stop):
            layer_to_ring[l] = ring_idx

    # ------------------------------------------------------------------ #
    # Step 1: get cell indices for every hit
    # ------------------------------------------------------------------ #
    layer_cell_idx_hit, phi_global_idx_hit, z_cell_idx_hit, valid = _hits_to_cell_indices_barrel(
        x, y, z, detector_patching_params
    )
   

    
    # print("layer_cell_idx", layer_cell_idx_hit[:10])
    # print("phi_global", phi_global_idx_hit[:10])
    # print("z_cell_idx", z_cell_idx_hit[:10])

    # ------------------------------------------------------------------ #
    # Step 2: map each hit to a patch
    # ------------------------------------------------------------------ #

    cells_per_patch_phi_hit = [cells_per_patch_phi[layer_i] for layer_i in layer_cell_idx_hit]
    # print("cells_per_patch_phi",cells_per_patch_phi_hit[:10])
    
    ring_patch_idx_hit   = [layer_to_ring[layer_i] for layer_i in layer_cell_idx_hit]
    z_patch_idx_hit     = z_cell_idx_hit  // cells_per_patch_z
    phi_patch_idx_hit = phi_global_idx_hit // cells_per_patch_phi_hit
    

    # print("ring_patch_idx_hit", ring_patch_idx_hit[:10])
    # print("z_patch_idx_hit", z_patch_idx_hit[:10])
    # print("phi_patch_idx_hit", phi_patch_idx_hit[:10])

    result: Dict[int, Dict] = {}

    # Pre-allocate cell tensors for every patch
    for p in registry["patches"]:
        ring_idx = p["r_idx"]
        l_start, l_stop = ring_to_layers[ring_idx]

        n_layers_in_ring = l_stop - l_start + 1
        cpp = cells_per_patch_phi[l_start]
        n_z = cells_per_patch_z

        result[p["patch_id"]] = {
            "cell_tensor" : np.zeros((n_layers_in_ring, cpp, n_z), dtype=np.float32),
            "true_coords": np.array([p["r_center"], p["phi_center"], p["z_center"]], dtype=np.float32),  # CHANGED
            "index_coords": np.array([p["r_idx"], p["phi_idx"], p["z_idx"]]),  # CHANGED
            "n_hits"      : 0,
        }

    # Scatter hits into cell tensors
    for hit_idx in np.where(valid)[0]:
        layer  = int(layer_cell_idx_hit[hit_idx])
        if layer not in layer_to_ring:
            continue
        ring_idx = ring_patch_idx_hit[hit_idx]
        l_start, l_stop = ring_to_layers[ring_idx]

        layer_local = layer - l_start
        phi_local      = phi_global_idx_hit[hit_idx]   %  cells_per_patch_phi[l_start]
        z_local =   z_cell_idx_hit[hit_idx] % cells_per_patch_z


        key = (ring_patch_idx_hit[hit_idx], phi_patch_idx_hit[hit_idx], z_patch_idx_hit[hit_idx])
      
        pid = registry["index"].get(key)
        if pid is None:
            print(f"No patch for key {key} – skipping hit")
            print(x[hit_idx], y[hit_idx], z[hit_idx])
            print(ring_patch_idx_hit[hit_idx], phi_global_idx_hit[hit_idx], z_cell_idx_hit[hit_idx])
            print()
            continue

        result[pid]["cell_tensor"][layer_local, phi_local, z_local] += energy[hit_idx]
        result[pid]["n_hits"] += 1


    # CHANGED: group by shape
    grouped = defaultdict(lambda: {
        "flat_tensor": [],
        "global_patch_ids": [],
        "local_patch_ids": [],
        "patch_positions": [],
    })
    
    for patch_id in result.keys():
        tensor = result[patch_id]["cell_tensor"]
        shape = tensor.shape  # (L, P, Z)
    
        grouped[shape]["flat_tensor"].append(tensor.reshape(-1))  # CHANGED: flatten here
        grouped[shape]["global_patch_ids"].append(patch_id)
        grouped[shape]["local_patch_ids"].append(result[patch_id]["index_coords"].reshape(-1))
        grouped[shape]["patch_positions"].append(result[patch_id]["true_coords"].reshape(-1))
    
    # stack
    for shape in grouped:
        grouped[shape]["flat_tensor"] = np.stack(grouped[shape]["flat_tensor"]) # shape: n_patches, n_cells_in_patch
        grouped[shape]["global_patch_ids"] = np.array(grouped[shape]["global_patch_ids"])
        grouped[shape]["local_patch_ids"] = np.array(grouped[shape]["local_patch_ids"])
        grouped[shape]["patch_positions"] = np.array(grouped[shape]["patch_positions"])
    
    return grouped

