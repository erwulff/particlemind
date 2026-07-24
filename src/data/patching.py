

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import math

"""
detector_vit_patching.py  (v6 — unified barrel + endcap)

Scheme
------
  side : "barrel"   → barrel_configs, symmetric z around 0.
         "positive" → endcap_configs, z > 0 (large positive z).
         "negative" → endcap_configs, z < 0 (mirror of positive endcap).
                      z_cell_idx = 0 always corresponds to the disc
                      closest to the IP.

  r   : layers grouped into rings via `groups`.
        Cell tensor per patch is a list of n_layers_in_ring 2-D arrays,
        each shape (phi_size_for_that_layer, cells_per_patch_z).
        phi_size varies by layer (Bresenham split of that layer's cpw).
  phi : 64 patches = patches_per_wedge (4) × num_wedges (16), wedge-aligned.
        Per-layer Bresenham split: patch i gets
            base + (1 if i < remainder else 0)  cells,
            base = cells_per_wedge[layer] // patches_per_wedge
            remainder = cells_per_wedge[layer] % patches_per_wedge
        Patch 0 always gets the larger slice, consistently across layers.
  z   : barrel  — continuous grid, grouped cells_per_patch_z at a time.
        endcap  — discrete z-disc layers (1 cell thick, large gaps between
                  them); cells_per_patch_z disc layers per z-patch.

Total patches = n_rings × n_phi_patches × n_z_patches.
"""

_PATCHES_PER_WEDGE = 4
_NUM_WEDGES        = 16
_N_PHI_PATCHES     = _PATCHES_PER_WEDGE * _NUM_WEDGES   # 64


# --------------------------------------------------------------------------- #
# Phi geometry helpers (per-layer, as in v3)
# --------------------------------------------------------------------------- #

def _phi_patch_sizes_for_cpw(
    cpw: int,
    patches_per_wedge: int = _PATCHES_PER_WEDGE,
) -> List[int]:
    """
    Example: cpw=23, patches=4  →  [6, 6, 6, 5]
             cpw=25, patches=4  →  [7, 6, 6, 6]
             cpw=45, patches=4  →  [12, 11, 11, 11]
    """
    base      = cpw // patches_per_wedge
    remainder = cpw %  patches_per_wedge
    return [base + (1 if i < remainder else 0) for i in range(patches_per_wedge)]


def _phi_patch_starts_for_cpw(
    cpw: int,
    patches_per_wedge: int = _PATCHES_PER_WEDGE,
) -> List[int]:
    """Starting phi-cell offset within a wedge for each within-wedge patch."""
    sizes  = _phi_patch_sizes_for_cpw(cpw, patches_per_wedge)
    starts = [0] * patches_per_wedge
    for i in range(1, patches_per_wedge):
        starts[i] = starts[i - 1] + sizes[i - 1]
    return starts


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def build_patch_registry(
    detector_patching_params: Dict,
    side: str          = "barrel",
    num_wedges:        int = _NUM_WEDGES,
    patches_per_wedge: int = _PATCHES_PER_WEDGE,
) -> Tuple[dict, dict, int]:
    """
    Build the complete patch registry for any sub-detector geometry.

    Parameters
    ----------
    side : "barrel"   — reads barrel_configs, symmetric z.
           "positive" — reads endcap_configs, z > 0.
           "negative" — reads endcap_configs, z < 0 (mirrored).

    Returns
    -------
    registry          : dict with full patch metadata
    unique_patch_sizes: dict keyed by (ring_idx, within_wedge_idx) ->
                        total number of cells in that patch shape
    total_patches     : int
    """
    assert side in ("barrel", "positive", "negative"), \
        f"side must be 'barrel', 'positive', or 'negative'; got {side!r}"

    cfg_key     = "barrel_configs" if side == "barrel" else "endcap_configs"
    cfg         = detector_patching_params[cfg_key]
    cell_size   = detector_patching_params["cell_size"]
    layer_width = detector_patching_params["layer_width"]

    cells_per_wedge   = cfg["cells_per_wedge"]
    cells_per_patch_z = cfg["cells_per_patch_z"]
    num_cells_z       = cfg["n_bins_z"]
    n_layers_total    = cfg["n_bins_x"]
    groups            = cfg.get("groups", cfg.get("group"))  # tolerate both spellings

    r_start = cfg["x_start_midpoint"] - layer_width / 2
    r_stop  = cfg["x_stop_midpoint"]  + layer_width / 2
    z_start = cfg["z_start_midpoint"] - cell_size   / 2
    z_stop  = cfg["z_stop_midpoint"]  + cell_size   / 2

    assert num_cells_z % cells_per_patch_z == 0, \
        "n_bins_z must be divisible by cells_per_patch_z"

    n_z_patches   = num_cells_z // cells_per_patch_z
    n_phi_patches = patches_per_wedge * num_wedges
    n_rings       = len(groups) - 1

    # ------------------------------------------------------------------ #
    # Ring <-> layer mappings
    # ------------------------------------------------------------------ #
    ring_to_layers: Dict[int, Tuple[int, int]] = {}
    layer_to_ring:  Dict[int, int]             = {}
    for ring_idx, (g_start, g_stop) in enumerate(zip(groups[:-1], groups[1:])):
        ring_to_layers[ring_idx] = (g_start, g_stop - 1)
        for l in range(g_start, g_stop):
            layer_to_ring[l] = ring_idx

    # ------------------------------------------------------------------ #
    # Per-layer phi patch geometry
    # ------------------------------------------------------------------ #
    layer_phi_sizes:  Dict[int, List[int]] = {
        l: _phi_patch_sizes_for_cpw(cells_per_wedge[l], patches_per_wedge)
        for l in range(n_layers_total)
    }
    layer_phi_starts: Dict[int, List[int]] = {
        l: _phi_patch_starts_for_cpw(cells_per_wedge[l], patches_per_wedge)
        for l in range(n_layers_total)
    }

    # ------------------------------------------------------------------ #
    # Continuous centre coordinates
    # ------------------------------------------------------------------ #
    r_edges   = np.linspace(r_start, r_stop, n_layers_total + 1)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2

    def _ring_r_center(ring_idx: int) -> float:
        l_start, l_stop = ring_to_layers[ring_idx]
        return float(np.mean(r_centers[l_start:l_stop + 1]))

    z_edges        = np.linspace(z_start, z_stop, num_cells_z + 1)
    z_cell_centers = (z_edges[:-1] + z_edges[1:]) / 2
    # Negative endcap: disc midpoints mirror the positive side.
    # z_cell_idx = 0 → disc closest to the IP (least negative z).
    if side == "negative":
        z_cell_centers = -z_cell_centers
    z_patch_centers = np.array([
        np.mean(z_cell_centers[j * cells_per_patch_z:(j + 1) * cells_per_patch_z])
        for j in range(n_z_patches)
    ])

    phi_patch_centers = np.array([
        2 * np.pi * (phi_idx + 0.5) / n_phi_patches
        for phi_idx in range(n_phi_patches)
    ])

    # ------------------------------------------------------------------ #
    # Build registry
    # ------------------------------------------------------------------ #
    registry = {
        "patches":            [],
        "index":              {},   # (ring_idx, phi_idx, z_idx) -> patch_id
        "n_rings":            n_rings,
        "n_phi_patches":      n_phi_patches,
        "n_z_patches":        n_z_patches,
        "cells_per_patch_z":  cells_per_patch_z,
        "patches_per_wedge":  patches_per_wedge,
        "num_wedges":         num_wedges,
        "ring_to_layers":     ring_to_layers,
        "layer_to_ring":      layer_to_ring,
        "layer_phi_sizes":    layer_phi_sizes,
        "layer_phi_starts":   layer_phi_starts,
        "side":               side,   # "barrel", "positive", or "negative"
    }

    patch_id = 0
    for ring_idx in range(n_rings):
        l_start, l_stop  = ring_to_layers[ring_idx]
        n_layers_in_ring = l_stop - l_start + 1
        rc               = _ring_r_center(ring_idx)

        for phi_idx in range(n_phi_patches):
            wedge_idx        = phi_idx // patches_per_wedge
            within_wedge_idx = phi_idx %  patches_per_wedge

            for z_idx in range(n_z_patches):
                info = {
                    "patch_id":          patch_id,
                    "ring_idx":          ring_idx,
                    "phi_idx":           phi_idx,
                    "z_idx":             z_idx,
                    "wedge_idx":         wedge_idx,
                    "within_wedge_idx":  within_wedge_idx,
                    "layer_start":       l_start,
                    "layer_stop":        l_stop,
                    "n_layers_in_ring":  n_layers_in_ring,
                    "r_center":          rc,
                    "phi_center":        float(phi_patch_centers[phi_idx]),
                    "z_center":          float(z_patch_centers[z_idx]),
                    "z_cell_start":      z_idx * cells_per_patch_z,
                    "z_cell_stop":       (z_idx + 1) * cells_per_patch_z - 1,
                }
                registry["patches"].append(info)
                registry["index"][(ring_idx, phi_idx, z_idx)] = patch_id
                patch_id += 1

    # ------------------------------------------------------------------ #
    # Unique patch sizes keyed by (ring_idx, within_wedge_idx)
    # ------------------------------------------------------------------ #
    unique_patch_sizes: Dict[Tuple[int, int], int] = {}
    for ring_idx in range(n_rings):
        l_start, l_stop  = ring_to_layers[ring_idx]
        n_layers_in_ring = l_stop - l_start + 1
        for ww in range(patches_per_wedge):
            total_cells = sum(
                layer_phi_sizes[l][ww] for l in range(l_start, l_stop + 1)
            ) * cells_per_patch_z
            unique_patch_sizes[(ring_idx, ww)] = total_cells

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"[Registry — {side}] {patch_id} total patches")
    print(f"           {n_rings} rings × {n_phi_patches} phi × {n_z_patches} z")
    print(f"           cells_per_patch_z = {cells_per_patch_z}")
    print(f"           sample phi patch sizes (within-wedge) per ring:")
    for r in range(n_rings):
        l_start, l_stop = ring_to_layers[r]
        sample_layer    = l_start
        print(f"             ring {r:2d} "
              f"(layers {l_start:2d}–{l_stop:2d}, "
              f"cpw[{sample_layer}]={cells_per_wedge[sample_layer]:3d}): "
              f"phi sizes (first layer) = {layer_phi_sizes[sample_layer]}")

    return registry, unique_patch_sizes, patch_id


# --------------------------------------------------------------------------- #
# Hit → cell indices
# --------------------------------------------------------------------------- #

def _hits_to_cell_indices(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    detector_patching_params: dict,
    side: str = "barrel",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    layer_idx            : radial layer index            (0 … n_layers-1)
    sector_idx           : wedge/sector index            (0 … num_wedges-1)
    within_wedge_phi_idx : phi-cell index within the wedge
                           (0 … cells_per_wedge[layer]-1)
    z_cell_idx           : longitudinal / disc layer index  (0 … n_bins_z-1)
    valid                : boolean mask of in-range hits

    For the negative endcap, hit z-values are negative; they are negated
    before binning so that z_cell_idx = 0 maps to the disc closest to the IP.
    """
    n_sectors  = 16
    half_wedge = np.pi / n_sectors

    cell_size   = detector_patching_params["cell_size"]
    layer_width = detector_patching_params["layer_width"]
    cfg_key     = "barrel_configs" if side == "barrel" else "endcap_configs"
    cfg         = detector_patching_params[cfg_key]

    n_bins_x = cfg["n_bins_x"]
    n_bins_y = cfg["n_bins_y"]
    n_bins_z = cfg["n_bins_z"]
    x_start  = cfg["x_start_midpoint"] - layer_width / 2
    x_stop   = cfg["x_stop_midpoint"]  + layer_width / 2
    y_start  = cfg["y_start_midpoint"] - cell_size   / 2
    y_stop   = cfg["y_stop_midpoint"]  + cell_size   / 2
    z_start  = cfg["z_start_midpoint"] - cell_size   / 2
    z_stop   = cfg["z_stop_midpoint"]  + cell_size   / 2

    cells_per_wedge = cfg["cells_per_wedge"]

    angles              = np.arctan2(y, x)
    angles[angles < 0] += 2 * np.pi
    angles_shifted      = (angles + half_wedge) % (2 * np.pi)
    sector_idx          = (angles_shifted / (2 * np.pi) * n_sectors).astype(int)

    delta_theta   = 2 * np.pi / n_sectors
    theta_center  = (sector_idx + 0.5) * delta_theta
    theta_center0 = 0.5 * delta_theta
    theta_rot     = -(theta_center - theta_center0)
    x_local = x * np.cos(theta_rot) - y * np.sin(theta_rot)
    y_local = x * np.sin(theta_rot) + y * np.cos(theta_rot)

    x_edges = np.linspace(x_start, x_stop, n_bins_x + 1)
    y_edges = np.linspace(y_start, y_stop, n_bins_y + 1)
    z_edges = np.linspace(z_start, z_stop, n_bins_z + 1)

    # Negative endcap: flip z so hits at z ≈ -z_start_mid map to z_cell_idx = 0.
    z_for_binning = -z if side == "negative" else z

    layer_idx  = np.digitize(x_local,        x_edges) - 1
    y_cell_idx = np.digitize(y_local,         y_edges) - 1
    z_cell_idx = np.digitize(z_for_binning,   z_edges) - 1

    max_layer  = max(cells_per_wedge.keys()) + 1
    # Clip before indexing offset_arr so out-of-range layer_idx doesn't crash;
    # the bad mask below will exclude those hits from the result.
    safe_layer = np.clip(layer_idx, 0, max_layer - 1)
    offset_arr = np.array([(n_bins_y - cells_per_wedge[i]) // 2 for i in range(max_layer)])
    within_wedge_phi_idx = y_cell_idx - offset_arr[safe_layer]

    bad = (
        (layer_idx            < 0) | (layer_idx            >= n_bins_x) |
        (y_cell_idx           < 0) | (y_cell_idx           >= n_bins_y) |
        (z_cell_idx           < 0) | (z_cell_idx           >= n_bins_z) |
        (within_wedge_phi_idx < 0)
    )
    if bad.any():
        print(f"  {bad.sum()} hits out of range — ignored")
        # print a few to debug
        for i in np.where(bad)[0][:5]:
            print(f"    hit {i}: x={x[i]:.2f}, y={y[i]:.2f}, z={z[i]:.2f}, "
                  f"layer_idx={layer_idx[i]}, y_cell_idx={y_cell_idx[i]}, "
                  f"within_wedge_phi_idx={within_wedge_phi_idx[i]}, "
                  f"z_cell_idx={z_cell_idx[i]}")

    return layer_idx, sector_idx, within_wedge_phi_idx, z_cell_idx, ~bad


# --------------------------------------------------------------------------- #
# Hit → patch assignment
# --------------------------------------------------------------------------- #

def assign_hits_to_patches(
    x:        np.ndarray,
    y:        np.ndarray,
    z:        np.ndarray,
    energy:   np.ndarray,
    registry: dict,
    detector_patching_params: dict,
) -> Dict:
    """
    Assign hits to ViT patches and return grouped per-patch data.

    Cell tensor layout
    ------------------
    Each patch stores a list of n_layers_in_ring 2-D arrays:
        cell_tensor[local_layer_idx] → np.ndarray (phi_size_for_layer, cells_per_patch_z)
    phi_size varies by layer (Bresenham split of that layer's cpw).

    Returns
    -------
    grouped : dict keyed by (ring_idx, within_wedge_idx), each value:
        {
          "flat_tensor"      : np.ndarray (n_patches, n_cells_in_patch)
          "global_patch_ids" : np.ndarray (n_patches,)
          "local_patch_ids"  : np.ndarray (n_patches, 3)  [ring, phi, z]
          "patch_positions"  : np.ndarray (n_patches, 3)  [r, phi, z]
        }
    """
    x      = np.asarray(x,      dtype=float)
    y      = np.asarray(y,      dtype=float)
    z      = np.asarray(z,      dtype=float)
    energy = np.asarray(energy, dtype=float)

    cells_per_patch_z = registry["cells_per_patch_z"]
    patches_per_wedge = registry["patches_per_wedge"]
    ring_to_layers    = registry["ring_to_layers"]
    layer_to_ring     = registry["layer_to_ring"]
    layer_phi_sizes   = registry["layer_phi_sizes"]
    layer_phi_starts  = registry["layer_phi_starts"]
    n_rings           = registry["n_rings"]
    n_layers_total    = max(layer_to_ring.keys()) + 1

    # ------------------------------------------------------------------ #
    # Step 1: cell indices
    # ------------------------------------------------------------------ #
    side = registry.get("side", "barrel")
    layer_idx, sector_idx, within_wedge_phi, z_cell_idx, valid = \
        _hits_to_cell_indices(x, y, z, detector_patching_params, side=side)

    # ------------------------------------------------------------------ #
    # Step 2: map hits to (ring_idx, phi_patch_idx, z_patch_idx)
    # ------------------------------------------------------------------ #
    hit_layer  = layer_idx[valid]
    hit_sector = sector_idx[valid]
    hit_ww_phi = within_wedge_phi[valid]
    hit_z_cell = z_cell_idx[valid]
    hit_energy = energy[valid]

    layer_to_ring_arr = np.array(
        [layer_to_ring[l] for l in range(n_layers_total)], dtype=int
    )
    hit_ring = layer_to_ring_arr[hit_layer]

    # starts_arr[layer, within_wedge_patch] — shape (n_layers, patches_per_wedge)
    starts_arr = np.array(
        [layer_phi_starts[l] for l in range(n_layers_total)], dtype=int
    )

    patch_starts = starts_arr[hit_layer]                          # (n_valid, patches_per_wedge)
    ww_patch_idx = np.sum(patch_starts <= hit_ww_phi[:, None], axis=1) - 1
    ww_patch_idx = np.clip(ww_patch_idx, 0, patches_per_wedge - 1)

    phi_patch_idx = hit_sector * patches_per_wedge + ww_patch_idx
    z_patch_idx   = hit_z_cell // cells_per_patch_z

    phi_local = hit_ww_phi - patch_starts[np.arange(len(hit_layer)), ww_patch_idx]
    z_local   = hit_z_cell % cells_per_patch_z

    ring_layer_starts_arr = np.array(
        [ring_to_layers[r][0] for r in range(n_rings)], dtype=int
    )
    layer_local = hit_layer - ring_layer_starts_arr[hit_ring]

    # ------------------------------------------------------------------ #
    # Step 3: pre-allocate result dict
    # ------------------------------------------------------------------ #
    result: Dict[int, Dict] = {}
    for p in registry["patches"]:
        pid              = p["patch_id"]
        ring_idx         = p["ring_idx"]
        ww               = p["within_wedge_idx"]
        l_start          = p["layer_start"]
        l_stop           = p["layer_stop"]
        n_layers_in_ring = p["n_layers_in_ring"]

        result[pid] = {
            "cell_tensor": [
                np.zeros((layer_phi_sizes[l][ww], cells_per_patch_z), dtype=np.float32)
                for l in range(l_start, l_stop + 1)
            ],
            "true_coords":  np.array(
                [p["r_center"], p["phi_center"], p["z_center"]], dtype=np.float32
            ),
            "index_coords": np.array(
                [p["ring_idx"], p["phi_idx"], p["z_idx"]], dtype=np.int32
            ),
            "n_hits":           0,
            "ring_idx":         ring_idx,
            "within_wedge_idx": ww,
        }

    # ------------------------------------------------------------------ #
    # Step 4: scatter
    # ------------------------------------------------------------------ #
    for i in range(len(hit_layer)):
        key = (int(hit_ring[i]), int(phi_patch_idx[i]), int(z_patch_idx[i]))
        pid = registry["index"].get(key)
        if pid is None:
            print(f"  No patch for key {key} — skipping hit")
            continue

        ll = int(layer_local[i])
        pl = int(phi_local[i])
        zl = int(z_local[i])

        max_pl = result[pid]["cell_tensor"][ll].shape[0] - 1
        if pl > max_pl:
            pl = max_pl

        result[pid]["cell_tensor"][ll][pl, zl] += hit_energy[i]
        result[pid]["n_hits"] += 1

    # ------------------------------------------------------------------ #
    # Step 5: group by (ring_idx, within_wedge_idx) and stack
    # ------------------------------------------------------------------ #
    grouped = defaultdict(lambda: {
        "flat_tensor":      [],
        "global_patch_ids": [],
        "local_patch_ids":  [],
        "patch_positions":  [],
    })

    for pid, data in result.items():
        key         = str((data["ring_idx"], data["within_wedge_idx"]))
        flat_tensor = np.concatenate([t.reshape(-1) for t in data["cell_tensor"]])

        grouped[key]["flat_tensor"].append(flat_tensor)
        grouped[key]["global_patch_ids"].append(pid)
        grouped[key]["local_patch_ids"].append(data["index_coords"])
        grouped[key]["patch_positions"].append(data["true_coords"])

    for key in grouped:
        grouped[key]["flat_tensor"]      = np.stack(grouped[key]["flat_tensor"])
        grouped[key]["global_patch_ids"] = np.array(grouped[key]["global_patch_ids"])
        grouped[key]["local_patch_ids"]  = np.stack(grouped[key]["local_patch_ids"])
        grouped[key]["patch_positions"]  = np.stack(grouped[key]["patch_positions"])


    return grouped

