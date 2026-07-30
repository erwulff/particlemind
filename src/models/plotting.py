import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch

# Region colours / labels shared across all patch plots
_REGION_COLORS = {"barrel": "#1f77b4", "endcap_pos": "#ff7f0e", "endcap_neg": "#2ca02c"}
_REGION_LABELS = {"barrel": "Barrel", "endcap_pos": "Endcap (+z)", "endcap_neg": "Endcap (−z)"}


def _region_from_key(key):
    """'barrel_(0, 1)' -> 'barrel'"""
    return key.split("_(")[0]


def _ring_from_key(key):
    """'barrel_(1, 2)' -> 1"""
    return int(key.split("_(")[1].split(",")[0].strip())


def _flat_to_2d(flat, cpz):
    """Reshape a flat patch tensor to (n_phi_total, cpz); fall back to column if not divisible."""
    if cpz > 1 and len(flat) % cpz == 0:
        return flat.reshape(-1, cpz)
    return flat.reshape(-1, 1)


def _axes_empty(ax):
    return not (ax.lines or ax.patches or ax.collections or ax.images or ax.texts or ax.artists or ax.tables)


def plot_model_hit(input_data, reco, labels, device="cuda", n_events_to_plot=2, n_scatterpoints_to_plot=200, masks=None, saveas=None):
    """Visualize the model.

    Parameters
    ----------
    model : nn.Module
        The model.
    samples : Tensor
        The input data.
    device : str, optional
        Device to use. The default is "cuda".
    n_examples_to_plot : int, optional
        Number of examples to plot. The default is 200.
    """

    vq_out = None
    # make empty axes invisible
    def is_axes_empty(ax):
        return not (ax.lines or ax.patches or ax.collections or ax.images or ax.texts or ax.artists or ax.tables)


    # input_data = input_data.to(device)
    # model = model.to(device)
   

    # # run the model on the input data
    # with torch.no_grad():
    #     # print(f"Model device: {next(model.parameters()).device}")
    #     # print(f"Samples device: {samples.device}")
    #     reco, vq_out, _ = model(None, input_data, masks)
       

        

    #     if vq_out is not None:
        
    #         master_z_q = vq_out["z_q"]
    #         master_z_e = vq_out["z"]
    #         master_idx = vq_out["q"]
       
    #         master_z_e = master_z_e.detach().cpu().numpy()
    #         master_z_q = master_z_q.detach().cpu().numpy()
    #         master_idx = master_idx.detach().cpu().numpy()

    input_data = input_data.detach().cpu().numpy()
    reco = reco.detach().cpu().numpy() # standardized in the model step


    

    labels = labels.detach().cpu().numpy()
    if masks is not None:
        masks = masks.detach().cpu().numpy()




    event_samples_E, event_samples_x, event_samples_y, event_samples_z = [], [], [], []
    reco_samples_E, reco_samples_x, reco_samples_y, reco_samples_z = [], [], [], []
    labels_event = []
    z_e, z_q, idx = [], [], []


    for event in range(n_events_to_plot):

        if masks is not None:
            mask = masks[event]
            event_samples_E.append(input_data[event, :, 3][mask == 1])
            event_samples_x.append(input_data[event, :, 0][mask == 1])
            event_samples_y.append(input_data[event, :, 1][mask == 1])
            event_samples_z.append(input_data[event, :, 2][mask == 1])
            reco_samples_E.append(reco[event, :, 3][mask == 1])
            reco_samples_x.append(reco[event, :, 0][mask == 1])
            reco_samples_y.append(reco[event, :, 1][mask == 1])
            reco_samples_z.append(reco[event, :, 2][mask == 1])
            labels_event.append(labels[event][mask == 1])
            if vq_out is not None:
                z_e.append(master_z_e[event].squeeze(1)[mask == 1])
                z_q.append(master_z_q[event].squeeze(1)[mask == 1])
                idx.append(master_idx[event].squeeze(1)[mask == 1])

        else:
            event_samples_E.append(input_data[event, :, 3])
            event_samples_x.append(input_data[event, :, 0])
            event_samples_y.append(input_data[event, :, 1])
            event_samples_z.append(input_data[event, :, 2])
            reco_samples_E.append(reco[event, :, 3])
            reco_samples_x.append(reco[event, :, 0])
            reco_samples_y.append(reco[event, :, 1])
            reco_samples_z.append(reco[event, :, 2])
            labels_event.append(labels[event])
            if vq_out is not None:
                z_e.append(master_z_e[event].squeeze(1))
                z_q.append(master_z_q[event].squeeze(1))
                idx.append(master_idx[event].squeeze(1))

    # concatenate all events
    event_samples_E_concat = np.concatenate(event_samples_E)
    event_samples_x_concat = np.concatenate(event_samples_x)
    event_samples_y_concat = np.concatenate(event_samples_y)
    event_samples_z_concat = np.concatenate(event_samples_z)
    
    reco_samples_E_concat = np.concatenate(reco_samples_E)
    reco_samples_x_concat = np.concatenate(reco_samples_x)
    reco_samples_y_concat = np.concatenate(reco_samples_y)
    reco_samples_z_concat = np.concatenate(reco_samples_z)
    labels_event_concat =  np.concatenate(labels_event)
    if vq_out is not None:
        z_e_concat = np.concatenate(z_e)
        z_q_concat = np.concatenate(z_q)
        idx_concat = np.concatenate(idx)

   


    #
    #
    # MULTI-EVENT FIGURES
    #
    #
    # create detached copy of the codebook to plot this
    fig, axarr = plt.subplots(1, 7, figsize=(7*7, 7))

    # histogram the energies
    ax = axarr[0]
    all_E = np.concatenate([event_samples_E_concat, reco_samples_E_concat])
    
    bins = np.linspace(np.min(all_E),  np.max(all_E), 50)
    ax.hist(event_samples_E_concat, bins=bins, label="samples", density=True, histtype="step", linewidth=2)
    ax.hist(reco_samples_E_concat, bins=bins, label="reco", density=True, histtype="step", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("$E$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")

    tmp = (event_samples_E_concat - reco_samples_E_concat)/event_samples_E_concat
    mm = (~np.isnan(tmp)) & (~np.isinf(tmp))

    # histogram the difference in energy
    ax = axarr[1]
    ax.hist(tmp[mm], bins=50, density=True, histtype="step", linewidth=2)
    ax.set_xlabel("$E_{true} - E_{reco}$ /$E_{true}$ ")
    ax.set_ylabel("Density")
    ax.set_yscale("log")

    if vq_out is not None:

        # scatter some zq - ze
        ax = axarr[2]
        ax.scatter(
            z_q_concat[:n_scatterpoints_to_plot, 0],
            z_q_concat[:n_scatterpoints_to_plot, 1],
            alpha=0.2,
            s=26,
            label="z_q",
        )
        ax.scatter(
            z_e_concat[:n_scatterpoints_to_plot, 0],
            z_e_concat[:n_scatterpoints_to_plot, 1],
            alpha=0.7,
            s=26,
            marker="x",
            label="z_e",
        )
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_1$")
        ax.set_title("Data space \nTrue vs reconstructed")
        ax.legend(loc="upper right")
    
        ax = axarr[3]
        ax.scatter(
            z_q_concat[:n_scatterpoints_to_plot, 0],
            z_q_concat[:n_scatterpoints_to_plot, 2],
            alpha=0.2,
            s=26,
            label="z_q",
        )
        ax.scatter(
            z_e_concat[:n_scatterpoints_to_plot, 0],
            z_e_concat[:n_scatterpoints_to_plot, 2],
            alpha=0.7,
            s=26,
            marker="x",
            label="z_e",
        )
        ax.set_xlabel("$x_0$")
        ax.set_ylabel("$x_2$")
        ax.set_title("Data space \nTrue vs reconstructed")
        ax.legend(loc="upper right")
        # plot the histogram of the codebook indices (i.e. a codebook_size x codebook_size
        # histogram with each entry in the histogram corresponding to one sample associated
        # with the corresponding codebook entry)
        ax = axarr[4]
        n_codes = model.vq_kwargs["num_codes"]
        bins = np.linspace(-0.5, n_codes + 0.5, n_codes + 1)
        ax.hist(idx_concat, bins=bins)
        ax.set_yscale("log")
        ax.set_title(
            "Codebook histogram\n(Each entry corresponds to one sample\nbeing associated with that" " codebook entry)",
            fontsize=8,
        )

    
    """
    ax = axarr[5]
        # Make a 3d scatter plot for event and reconstructed samples in x, y, z
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    
    cmap = plt.get_cmap("gist_ncar")
    colors = cmap(np.linspace(0, 1, len(unique_labels)))

    
    ax = fig.add_subplot(1, 7,  6, projection='3d')
    # plot event (true) samples, color-coded by label
    hit_clusters_true, hit_clusters_reco = [], []
    for i, label in enumerate(unique_labels):
        mask = labels_event == label
        ax.scatter(
            event_samples_x[mask],
            event_samples_y[mask],
            event_samples_z[mask],
            s=60,
            alpha=0.8,
            color=colors[i],
            edgecolor="black",
            linewidth=0.6,
            marker="o",
            label=f"label {label} (true)",
        )

        ax.scatter(
            reco_samples_x[mask],
            reco_samples_y[mask],
            reco_samples_z[mask],
            s=100,
            alpha=0.9,
            color="none",          # hollow, improves visibility
            edgecolor=colors[i],
            linewidth=1.5,
            marker="o",
            label=f"label {label} (reco)",
        )


    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    #ax.legend(loc=(1,0))
    ax.set_title("Hit spatial distribution")
    """


    # # resolution (cluster energy)
    ax = axarr[5]
    unique_labels = [np.unique(l) for l in labels_event]
    hit_clusters_true, hit_clusters_reco = [], []
    
    for event_i, labels_event_i in enumerate(labels_event):
        for unique_label_event_i in unique_labels[event_i]:
            mask_event_i_label_i = labels_event_i == unique_label_event_i
            hit_clusters_true.append(np.sum(event_samples_E[event_i][mask_event_i_label_i]))
            hit_clusters_reco.append(np.sum(reco_samples_E[event_i][mask_event_i_label_i]))

    tmp = (np.array(hit_clusters_true) - np.array(hit_clusters_reco))/np.array(hit_clusters_true)
    mm = (~np.isnan(tmp)) & (~np.isinf(tmp))

    ax.hist(tmp[mm], bins=50, density=True, histtype="step", linewidth=2)
    ax.set_xlabel( "$E_{true} - E_{reco}$  / $E_{true}$ per cluster")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.legend(loc="upper right")


    
    for ax in axarr.flatten():
        if is_axes_empty(ax):
            ax.set_visible(False)


    fig.tight_layout()
    plt.show()
    if saveas is not None:
        fig.savefig(saveas+"_multi_event_figures.png")

    #
    #
    # SINGLE EVENT FIGURES
    #
    #

     # pull the first event for scatter plots
    mask = masks[0]
    single_event_samples_x = input_data[0, :, 0][mask == 1]
    single_event_samples_y = input_data[0, :, 1][mask == 1]
    single_event_samples_z = input_data[0, :, 2][mask == 1]
    single_reco_samples_x = reco[0, :, 0][mask == 1]
    single_reco_samples_y = reco[0, :, 1][mask == 1]
    single_reco_samples_z = reco[0, :, 2][mask == 1]

    
    all_x = np.concatenate([single_event_samples_x, single_reco_samples_x])
    all_y = np.concatenate([single_event_samples_y, single_reco_samples_y])
    all_z = np.concatenate([single_event_samples_z, single_reco_samples_z])

    bins_x = np.linspace(np.min(all_x), np.max(all_x), 100)
    bins_y = np.linspace(np.min(all_y), np.max(all_y), 100)
    bins_z = np.linspace(np.min(all_z), np.max(all_z), 100)

    fig, axarr = plt.subplots(1, 6, figsize=(7*6, 6))

    # data, x-y
    ax = axarr[0]
    #print(single_event_samples_x)
    #print(single_event_samples_y)
    h = ax.hist2d(single_event_samples_x, single_event_samples_y, bins=[bins_x, bins_y], norm="log", density=True)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Data")
    # add colorbar axis
    if np.isfinite(h[0]).any() and np.nanmin(h[0]) < np.nanmax(h[0]):
        plt.colorbar(h[3], ax=ax)

    # reco, x-y
    ax = axarr[1]
    h = ax.hist2d(single_reco_samples_x, single_reco_samples_y, bins=[bins_x, bins_y], norm="log", density=True)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Reco")
    if np.isfinite(h[0]).any() and np.nanmin(h[0]) < np.nanmax(h[0]):
        plt.colorbar(h[3], ax=ax)

    # data, x-z
    ax = axarr[2]
    h = ax.hist2d(single_event_samples_x, single_event_samples_z, bins=[bins_x, bins_z], norm="log", density=True)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_title("Data")
    if np.isfinite(h[0]).any() and np.nanmin(h[0]) < np.nanmax(h[0]):
        plt.colorbar(h[3], ax=ax)

    # reco, x-z
    ax = axarr[3]
    h = ax.hist2d(single_reco_samples_x, single_reco_samples_z, bins=[bins_x, bins_z], norm="log", density=True)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_title("Reco")
    if np.isfinite(h[0]).any() and np.nanmin(h[0]) < np.nanmax(h[0]):
        plt.colorbar(h[3], ax=ax)

    # data, y-z
    ax = axarr[4]
    h = ax.hist2d(single_event_samples_y, single_event_samples_z, bins=[bins_y, bins_z], norm="log", density=True)
    ax.set_xlabel("$y$")
    ax.set_ylabel("$z$")
    ax.set_title("Data")
    if np.isfinite(h[0]).any() and np.nanmin(h[0]) < np.nanmax(h[0]):
        plt.colorbar(h[3], ax=ax)

    # reco, y-z
    ax = axarr[5]
    h = ax.hist2d(single_reco_samples_y, single_reco_samples_z, bins=[bins_y, bins_z], norm="log", density=True)
    ax.set_xlabel("$y$")
    ax.set_ylabel("$z$")
    ax.set_title("Reco")
    if np.isfinite(h[0]).any() and np.nanmin(h[0]) < np.nanmax(h[0]):
        plt.colorbar(h[3], ax=ax)


    for ax in axarr.flatten():
        if is_axes_empty(ax):
            ax.set_visible(False)


    fig.tight_layout()
    plt.show()
    if saveas is not None:
        fig.savefig(saveas+"_single_event_figures.png")

        
        
def plot_model_patch(
    batch,
    patches_chunked_reco,
    vq_out,
    num_codes,
    device="cuda",
    n_scatterpoints_to_plot=300,
    # cells_per_patch_z for the 2-D patch-interior images.
    # For HCAL: barrel=11, endcap=1.  Change for ECAL (barrel=57, endcap=1).
    cells_per_patch_z_barrel=11,
    cells_per_patch_z_endcap=1,
    saveas=None,
):
    # ------------------------------------------------------------------ #
    # Unpack VQ tensors (safe against extra trailing dim)
    # ------------------------------------------------------------------ #
    if vq_out is not None:
        master_z_q = vq_out["z_q"].detach().cpu().numpy()
        master_z_e = vq_out["z"].detach().cpu().numpy()
        master_idx = vq_out["q"].detach().cpu().numpy()
        if master_idx.ndim == 3:
            master_idx = master_idx.squeeze(2)
        if master_z_q.ndim == 4:
            master_z_q = master_z_q.squeeze(2)
        if master_z_e.ndim == 4:
            master_z_e = master_z_e.squeeze(2)

    B = next(iter(batch.values()))["flat_tensor"].shape[0]

    # gid_region[b][gid] = region string — used to colour latent-space points
    gid_region = [{} for _ in range(B)]
    for key in batch.keys():
        region = _region_from_key(key)
        for b in range(B):
            for gid in batch[key]["global_patch_ids"][b].cpu().numpy().tolist():
                gid_region[b][int(gid)] = region

    # ------------------------------------------------------------------ #
    # Multi-event energy data, split by region and by ring
    # ------------------------------------------------------------------ #
    region_E_true   = defaultdict(list)
    region_E_reco   = defaultdict(list)
    # region_ring_res[region][ring_idx] = list of (E_r-E_t)/E_t values
    region_ring_res = defaultdict(lambda: defaultdict(list))

    for key in batch.keys():
        region   = _region_from_key(key)
        ring_idx = _ring_from_key(key)
        ft = batch[key]["flat_tensor"].detach().cpu().numpy()     # (B, P_k, C)
        fr = patches_chunked_reco[key].detach().cpu().numpy()
        mk = batch[key]["mask"].detach().cpu().numpy()            # (B, P_k)

        E_t = ft.sum(axis=2)
        E_r = fr.sum(axis=2)
        for b in range(B):
            m  = mk[b] > 0
            et = E_t[b][m];  er = E_r[b][m]
            region_E_true[region].extend(et.tolist())
            region_E_reco[region].extend(er.tolist())
            nz = et > 0
            if nz.any():
                region_ring_res[region][ring_idx].extend(
                    ((er[nz] - et[nz]) / et[nz]).tolist()
                )

    region_E_true = {k: np.array(v) for k, v in region_E_true.items()}
    region_E_reco = {k: np.array(v) for k, v in region_E_reco.items()}

    # Split latent vectors and codebook indices by region
    if vq_out is not None:
        region_z_q   = defaultdict(list)
        region_z_e   = defaultdict(list)
        region_codes = defaultdict(list)

        for b in range(B):
            # model argsorts patches by ascending global_patch_id before the transformer
            all_gids = []
            for key in batch.keys():
                all_gids.extend(batch[key]["global_patch_ids"][b].cpu().numpy().tolist())
            all_gids   = np.array(all_gids)
            sort_order = np.argsort(all_gids)

            for rank, orig_idx in enumerate(sort_order):
                gid    = int(all_gids[orig_idx])
                region = gid_region[b].get(gid, "unknown")
                region_z_q[region].append(master_z_q[b][rank])
                region_z_e[region].append(master_z_e[b][rank])
                region_codes[region].append(int(master_idx[b][rank]))

        for r in list(region_z_q.keys()):
            region_z_q[r]   = np.array(region_z_q[r])
            region_z_e[r]   = np.array(region_z_e[r])
            region_codes[r] = np.array(region_codes[r])

    # ------------------------------------------------------------------ #
    # Single-event (event 0) spatial data, split by region
    # ------------------------------------------------------------------ #
    single = defaultdict(lambda: defaultdict(list))

    for key in batch.keys():
        region = _region_from_key(key)
        pos = batch[key]["patch_positions"][0].detach().cpu().numpy()  # (P_k, 3): [r, phi, z]
        ft  = batch[key]["flat_tensor"][0].detach().cpu().numpy()
        fr  = patches_chunked_reco[key][0].detach().cpu().numpy()
        mk  = batch[key]["mask"][0].detach().cpu().numpy()

        m   = mk > 0
        r_arr, phi_arr, z_arr = pos[m, 0], pos[m, 1], pos[m, 2]
        E_t = ft[m].sum(axis=1)
        E_r = fr[m].sum(axis=1)
        res = (E_r - E_t) / (E_t + 1e-8)

        single[region]["r"].extend(r_arr.tolist())
        single[region]["phi"].extend(phi_arr.tolist())
        single[region]["z"].extend(z_arr.tolist())
        single[region]["x"].extend((r_arr * np.cos(phi_arr)).tolist())
        single[region]["y"].extend((r_arr * np.sin(phi_arr)).tolist())
        single[region]["E_true"].extend(E_t.tolist())
        single[region]["E_reco"].extend(E_r.tolist())
        single[region]["res"].extend(res.tolist())

    # Attach codebook index per patch for the single event
    if vq_out is not None:
        all_gids_ev0 = []
        for key in batch.keys():
            all_gids_ev0.extend(
                batch[key]["global_patch_ids"][0].cpu().numpy().tolist()
            )
        all_gids_ev0 = np.array(all_gids_ev0)
        sort_ev0     = np.argsort(all_gids_ev0)
        gid_to_code  = {int(all_gids_ev0[orig]): int(master_idx[0][rank])
                        for rank, orig in enumerate(sort_ev0)}
        for key in batch.keys():
            region = _region_from_key(key)
            gids   = batch[key]["global_patch_ids"][0].cpu().numpy()
            mk     = batch[key]["mask"][0].cpu().numpy()
            single[region]["codes"].extend(
                [gid_to_code[int(g)] for g in gids[mk > 0]]
            )

    for region in single:
        for k in single[region]:
            single[region][k] = np.array(single[region][k])

    # shared residual colour scale (99th-percentile robust)
    all_res = np.concatenate(
        [single[r]["res"] for r in single if len(single[r].get("res", [])) > 0]
    )
    vmax_res = float(np.percentile(np.abs(all_res), 99)) if len(all_res) else 1.0
    vmin_res = -vmax_res

    # ================================================================== #
    # FIGURE 1 — Multi-event summary
    # Panels: per-region ΔE/E hists | E_reco vs E_true log-log scatter |
    #         per-ring violin plots | (if VQ) latent scatter × 2 + codebook usage
    # ================================================================== #
    n_rows = 2 if vq_out is not None else 1
    fig1, axarr1 = plt.subplots(n_rows, 3, figsize=(21, 7 * n_rows))
    if n_rows == 1:
        axarr1 = axarr1.reshape(1, 3)

    # [0,0] Per-region ΔE/E histograms (all batch events)
    ax = axarr1[0, 0]
    for region in ["barrel", "endcap_pos", "endcap_neg"]:
        et = region_E_true.get(region, np.array([]))
        er = region_E_reco.get(region, np.array([]))
        nz = et > 0
        if nz.any():
            res = (er[nz] - et[nz]) / et[nz]
            ax.hist(res, bins=100, histtype="step", linewidth=2,
                    color=_REGION_COLORS[region], label=_REGION_LABELS[region], density=True)
    ax.set_xlabel(r"$(E_{reco} - E_{true}) / E_{true}$")
    ax.set_ylabel("Density"); ax.set_yscale("log")
    ax.set_title("Per-region energy resolution"); ax.legend()

    # [0,1] E_reco vs E_true log-log scatter (up to 500 pts per region)
    ax = axarr1[0, 1]
    for region in ["barrel", "endcap_pos", "endcap_neg"]:
        et = region_E_true.get(region, np.array([]))
        er = region_E_reco.get(region, np.array([]))
        valid = (et > 0) & (er > 0)
        if valid.any():
            idx = np.random.choice(np.where(valid)[0],
                                   size=min(500, valid.sum()), replace=False)
            ax.scatter(et[idx], er[idx], s=8, alpha=0.4,
                       color=_REGION_COLORS[region], label=_REGION_LABELS[region])
    all_pos = np.concatenate([v[v > 0] for v in region_E_true.values() if (v > 0).any()] or [np.array([1])])
    lims = [all_pos.min(), all_pos.max()]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$E_{true}$ [MeV]"); ax.set_ylabel(r"$E_{reco}$ [MeV]")
    ax.set_title(r"$E_{reco}$ vs $E_{true}$ per patch"); ax.legend(markerscale=2, fontsize=8)

    # [0,2] Per-ring ΔE/E violin plots, coloured by region
    ax = axarr1[0, 2]
    vdata, vlabels, vcolors = [], [], []
    for region in ["barrel", "endcap_pos", "endcap_neg"]:
        short = {"barrel": "B", "endcap_pos": "E+", "endcap_neg": "E−"}[region]
        for ring_idx in sorted(region_ring_res.get(region, {}).keys()):
            data = region_ring_res[region][ring_idx]
            if len(data) > 1:
                vdata.append(data); vlabels.append(f"{short}.{ring_idx}")
                vcolors.append(_REGION_COLORS[region])
    if vdata:
        pos   = list(range(len(vdata)))
        parts = ax.violinplot(vdata, positions=pos, showmedians=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(vcolors[i]); pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        ax.set_xticks(pos); ax.set_xticklabels(vlabels, fontsize=8)
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.set_ylabel(r"$(E_{reco} - E_{true}) / E_{true}$")
        ax.set_title("Resolution by ring")
        ax.legend(handles=[Patch(facecolor=_REGION_COLORS[r], alpha=0.6,
                                 label=_REGION_LABELS[r])
                            for r in ["barrel", "endcap_pos", "endcap_neg"]
                            if r in region_ring_res], fontsize=8)

    # [1,0–1] Latent space z_q scatter coloured by region (VQ only)
    if vq_out is not None:
        for col, (d0, d1) in enumerate([(0, 1), (0, 2)]):
            ax = axarr1[1, col]
            for region in ["barrel", "endcap_pos", "endcap_neg"]:
                if region not in region_z_q or len(region_z_q[region]) == 0:
                    continue
                zq     = region_z_q[region]
                n_show = min(n_scatterpoints_to_plot, len(zq))
                ax.scatter(zq[:n_show, d0], zq[:n_show, d1], s=12, alpha=0.4,
                           color=_REGION_COLORS[region], label=_REGION_LABELS[region])
            ax.set_xlabel(f"$z_{d0}$"); ax.set_ylabel(f"$z_{d1}$")
            ax.set_title(f"Latent $z_q$ (dims {d0},{d1}) by region"); ax.legend(fontsize=8)

        # [1,2] Codebook usage histogram, one curve per region
        if num_codes is not None:
            ax = axarr1[1, 2]
            bins = np.arange(-0.5, num_codes + 0.5, 1)
            for region in ["barrel", "endcap_pos", "endcap_neg"]:
                if region in region_codes and len(region_codes[region]) > 0:
                    ax.hist(region_codes[region], bins=bins, histtype="step", linewidth=2,
                            color=_REGION_COLORS[region], label=_REGION_LABELS[region])
            ax.set_yscale("log")
            ax.set_xlabel("Codebook index"); ax.set_ylabel("Count")
            ax.set_title("Codebook usage by region"); ax.legend(fontsize=8)

    for ax in axarr1.flatten():
        if _axes_empty(ax):
            ax.set_visible(False)
    fig1.tight_layout()
    if saveas is not None:
        fig1.savefig(saveas + "_multi_event.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ================================================================== #
    # FIGURE 2 — Single-event detector views
    # Panels: full r-z | barrel x-y | endcap+z x-y | endcap-z x-y |
    #         (if VQ) r-z coloured by codebook index
    # ================================================================== #
    n_panels = 5 if vq_out is not None else 4
    fig2, axarr2 = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))

    # [0] Full-detector r-z slice: barrel forms horizontal band, endcaps form vertical caps
    ax = axarr2[0]
    sc_rz = None
    for region in ["barrel", "endcap_pos", "endcap_neg"]:
        d  = single.get(region, {})
        et = d.get("E_true", np.array([]))
        nz = et > 0
        if not nz.any():
            continue
        sizes = 10 + 50 * (et[nz] / et[nz].max()) ** 0.4   # size ∝ sqrt(E_true)
        sc_rz = ax.scatter(d["z"][nz], d["r"][nz],
                           c=d["res"][nz], s=sizes,
                           cmap="coolwarm", vmin=vmin_res, vmax=vmax_res,
                           alpha=0.75, edgecolors="none")
    if sc_rz is not None:
        fig2.colorbar(sc_rz, ax=ax, label=r"$(E_{reco}-E_{true})/E_{true}$")
    ax.set_xlabel("$z$ [mm]"); ax.set_ylabel("$r$ [mm]")
    ax.set_title("Full detector $r$–$z$\n(size ∝ $E_{true}$, colour = residual)")

    # [1–3] Transverse x-y views — barrel and both endcap discs
    for panel_idx, region in enumerate(["barrel", "endcap_pos", "endcap_neg"], start=1):
        ax  = axarr2[panel_idx]
        d   = single.get(region, {})
        et  = d.get("E_true", np.array([]))
        nz  = et > 0
        if nz.any():
            sc = ax.scatter(d["x"][nz], d["y"][nz], c=d["res"][nz], s=15,
                            cmap="coolwarm", vmin=vmin_res, vmax=vmax_res, alpha=0.7)
            fig2.colorbar(sc, ax=ax)
        ax.set_xlabel("$x$ [mm]"); ax.set_ylabel("$y$ [mm]")
        ax.set_title(f"{_REGION_LABELS[region]}: transverse ($x$–$y$)\ncolour = residual")
        ax.set_aspect("equal")

    # [4] r-z coloured by codebook index (VQ only): reveals geometry-aware codes
    if vq_out is not None:
        ax = axarr2[4]
        all_z4, all_r4, all_c4 = [], [], []
        for region in ["barrel", "endcap_pos", "endcap_neg"]:
            d  = single.get(region, {})
            et = d.get("E_true", np.array([]))
            nz = et > 0
            if "codes" in d and nz.any():
                all_z4.extend(d["z"][nz].tolist())
                all_r4.extend(d["r"][nz].tolist())
                all_c4.extend(d["codes"][nz].tolist())
        if all_z4:
            sc4 = ax.scatter(all_z4, all_r4, c=all_c4, s=12, alpha=0.75,
                             cmap="tab20", vmin=0, vmax=num_codes)
            fig2.colorbar(sc4, ax=ax, label="Codebook index")
        ax.set_xlabel("$z$ [mm]"); ax.set_ylabel("$r$ [mm]")
        ax.set_title("$r$–$z$: codebook index")

    for ax in axarr2.flatten():
        if _axes_empty(ax):
            ax.set_visible(False)
    fig2.tight_layout()
    if saveas is not None:
        fig2.savefig(saveas + "_single_event.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ================================================================== #
    # FIGURE 3 — Patch interior (event 0)
    # For each region: 2 noisiest (highest E_true) + 2 quietest (lower-half non-zero).
    # Barrel patches → 2-D imshow of shape (n_phi_total, cpz).
    # Endcap patches → 1-D bar chart (cpz=1, so no z-spread within patch).
    # Each row: True | Reco | Reco − True
    # ================================================================== #
    N_NOISY, N_QUIET = 2, 2

    # Gather all valid patches for event 0, keyed by region
    raw_patches = defaultdict(list)
    for key in batch.keys():
        region   = _region_from_key(key)
        ring_idx = _ring_from_key(key)
        cpz      = cells_per_patch_z_barrel if region == "barrel" else cells_per_patch_z_endcap
        ft = batch[key]["flat_tensor"][0].detach().cpu().numpy()
        fr = patches_chunked_reco[key][0].detach().cpu().numpy()
        mk = batch[key]["mask"][0].detach().cpu().numpy()
        ps = batch[key]["patch_positions"][0].detach().cpu().numpy()
        for i in range(ft.shape[0]):
            if mk[i] > 0:
                raw_patches[region].append({
                    "flat_true": ft[i], "flat_reco": fr[i],
                    "E_true": float(ft[i].sum()), "position": ps[i],
                    "ring_idx": ring_idx, "cpz": cpz,
                })

    # Select noisy (highest energy) + quiet (from lower half of non-zero patches)
    chosen_patches = {}
    for region, patches in raw_patches.items():
        sorted_p = sorted(patches, key=lambda p: p["E_true"], reverse=True)
        nonzero  = [p for p in sorted_p if p["E_true"] > 0]
        noisy    = nonzero[:N_NOISY]
        lower    = nonzero[len(nonzero) // 2:]
        if len(lower) >= N_QUIET:
            idx   = np.round(np.linspace(0, len(lower) - 1, N_QUIET)).astype(int)
            quiet = [lower[int(i)] for i in idx]
        else:
            quiet = lower[:N_QUIET]
        for j, p in enumerate(noisy + quiet):
            p["label"] = f"noisy {j + 1}" if j < N_NOISY else f"quiet {j - N_NOISY + 1}"
        chosen_patches[region] = noisy + quiet

    total_rows = sum(len(v) for v in chosen_patches.values())
    if total_rows == 0:
        return

    fig3, axarr3 = plt.subplots(total_rows, 3, figsize=(15, 4 * total_rows), squeeze=False)

    row = 0
    for region in ["barrel", "endcap_pos", "endcap_neg"]:
        if region not in chosen_patches:
            continue
        for patch in chosen_patches[region]:
            ft   = patch["flat_true"]
            fr   = patch["flat_reco"]
            diff = fr - ft
            cpz  = patch["cpz"]
            pos  = patch["position"]
            row_title = (
                f"{_REGION_LABELS[region]} | Ring {patch['ring_idx']} | {patch['label']}\n"
                f"r={pos[0]:.0f} mm, φ={pos[1]:.2f} rad, z={pos[2]:.0f} mm"
                f" | $E_{{true}}$={patch['E_true']:.2f} MeV"
            )

            if cpz <= 1:
                # Endcap: one z-disc per patch → 1-D bar chart over φ cells
                x_idx = np.arange(len(ft))
                for col, (data, sub) in enumerate([
                    (ft, "True"), (fr, "Reco"), (diff, "Reco − True")
                ]):
                    ax = axarr3[row, col]
                    if col < 2:
                        ax.bar(x_idx, data, color=_REGION_COLORS[region], alpha=0.7)
                    else:
                        ax.bar(x_idx, data,
                               color=["#d62728" if d > 0 else "#1f77b4" for d in data],
                               alpha=0.7)
                        ax.axhline(0, color="black", lw=0.6)
                    ax.set_xlabel("φ cell index"); ax.set_ylabel("Energy [MeV]")
                    ax.set_title(f"{row_title}\n{sub}" if col == 0 else sub, fontsize=7)
            else:
                # Barrel: 2-D heatmap of shape (n_phi_total, cpz)
                # x-axis = z cell within patch, y-axis = stacked φ cells across all layers
                img_t = _flat_to_2d(ft,   cpz)
                img_r = _flat_to_2d(fr,   cpz)
                img_d = _flat_to_2d(diff, cpz)
                vmax_p   = max(img_t.max(), img_r.max(), 1e-9)
                vmax_dif = max(np.abs(img_d).max(), 1e-9)
                for col, (img, sub, cmap, vlo, vhi) in enumerate([
                    (img_t, "True",        "hot", 0,       vmax_p),
                    (img_r, "Reco",        "hot", 0,       vmax_p),
                    (img_d, "Reco − True", "bwr", -vmax_dif, vmax_dif),
                ]):
                    ax = axarr3[row, col]
                    im = ax.imshow(img, aspect="auto", cmap=cmap,
                                   vmin=vlo, vmax=vhi,
                                   interpolation="nearest", origin="lower")
                    fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_xlabel("$z$ cell index")
                    ax.set_ylabel("φ cell (layers stacked)")
                    ax.set_title(f"{row_title}\n{sub}" if col == 0 else sub, fontsize=7)
            row += 1

    fig3.tight_layout()
    if saveas is not None:
        fig3.savefig(saveas + "_patch_interior.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)



def plot_loss(loss_history, lr_history, moving_average=100):
    if len(loss_history) < moving_average:
        print("Not enough steps to plot loss history")
        return
    fig, ax1 = plt.subplots(figsize=(5, 2))
    ax2 = ax1.twinx()

    # Plot loss history
    loss_history = np.array(loss_history)
    loss_history = np.convolve(loss_history, np.ones(moving_average), "valid") / moving_average
    ax1.plot(loss_history, color="blue")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.set_title(f"Loss history (moving average over {moving_average} steps)", fontsize=8)

    # Plot lr history
    ax2.plot(lr_history, color="red")
    ax2.set_ylabel("Learning Rate")

    fig.tight_layout()
    plt.show()