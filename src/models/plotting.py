import numpy as np
import torch
import matplotlib.pyplot as plt

 

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
    # event_samples_x_concat = np.concatenate(event_samples_x)
    # event_samples_y_concat = np.concatenate(event_samples_y)
    # event_samples_z_concat = np.concatenate(event_samples_z)
    
    reco_samples_E_concat = np.concatenate(reco_samples_E)
    # reco_samples_x_concat = np.concatenate(reco_samples_x)
    # reco_samples_y_concat = np.concatenate(reco_samples_y)
    # reco_samples_z_concat = np.concatenate(reco_samples_z)
    # labels_event_concat =  np.concatenate(labels_event)
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


    # resolution (cluster energy)
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
    #ax.legend(loc="upper right")


    
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

        
        
def plot_model_patch(batch, patches_chunked_reco, vq_out, num_codes, device="cuda", n_scatterpoints_to_plot=300, saveas=None):

    def is_axes_empty(ax):
        return not (ax.lines or ax.patches or ax.collections or ax.images or ax.texts or ax.artists or ax.tables)

    # -----------------------------
    # LATENT + CODEBOOK (UNCHANGED)
    # -----------------------------
    if vq_out is not None:
        master_z_q = vq_out["z_q"].squeeze(2).detach().cpu().numpy() # (B, P, LATENT_DIM)
        master_z_e = vq_out["z"].squeeze(2).detach().cpu().numpy() # (B, P, LATENT_DIM)
        master_idx = vq_out["q"].squeeze(2).detach().cpu().numpy() # (B, P) 
    
    
    
        # flatten across all batches
        z_q_concat = np.concatenate([master_z_q[i] for i in range(len(master_z_q))]) # (B*P, LATENT_DIM)
        z_e_concat = np.concatenate([master_z_e[i] for i in range(len(master_z_e))]) # (B*P, LATENT_DIM)
        idx_concat = np.concatenate([master_idx[i] for i in range(len(master_idx))]) # (B*P)


    # ✅ CHANGED: now 4 panels (added resolution)
    fig, axarr = plt.subplots(1, 4, figsize=(7*4, 7))  # CHANGED

    if vq_out is not None:
    
        # scatter z_q vs z_e
        ax = axarr[0]
        ind0, ind1 = 0, 1
        ax.scatter(z_q_concat[:n_scatterpoints_to_plot, ind0],
                   z_q_concat[:n_scatterpoints_to_plot, ind1],
                   alpha=0.2, s=26, label="z_q")
        ax.scatter(z_e_concat[:n_scatterpoints_to_plot, ind0],
                   z_e_concat[:n_scatterpoints_to_plot, ind1],
                   alpha=0.7, s=26, marker="x", label="z_e")
        ax.set_xlabel(f"$x_{ind0}$")
        ax.set_ylabel(f"$x_{ind1}$")
        ax.set_title("Latent space: z_q vs z_e")
        ax.legend()
    
        ax = axarr[1]
        ind0, ind1 = 0, 2
        ax.scatter(z_q_concat[:n_scatterpoints_to_plot, ind0],
                   z_q_concat[:n_scatterpoints_to_plot, ind1],
                   alpha=0.2, s=26, label="z_q")
        ax.scatter(z_e_concat[:n_scatterpoints_to_plot, ind0],
                   z_e_concat[:n_scatterpoints_to_plot, ind1],
                   alpha=0.7, s=26, marker="x", label="z_e")
        ax.set_xlabel(f"$x_{ind0}$")
        ax.set_ylabel(f"$x_{ind1}$")
        ax.set_title("Latent space: z_q vs z_e")
        ax.legend()
        
        # codebook usage
        ax = axarr[2]
    
        bins = np.linspace(-0.5, num_codes + 0.5, num_codes + 1)
        ax.hist(idx_concat, bins=bins)
        ax.set_yscale("log")
        ax.set_title("Codebook usage")

    # ---------------------------------------
    # NEW: MULTI-EVENT ENERGY RESOLUTION
    # ---------------------------------------
    E_true_all, E_reco_all = {}, {}  # NEW

    for key in batch.keys():  # NEW
        flat_true = batch[key]["flat_tensor"].detach().cpu().numpy()
        flat_reco = patches_chunked_reco[key].detach().cpu().numpy()

        E_true = flat_true.sum(axis=2)
        E_reco = flat_reco.sum(axis=2)

        E_true_all[key] = np.nan_to_num(E_true.reshape(-1))
        E_reco_all[key] = np.nan_to_num(E_reco.reshape(-1))

 

    

    ax = axarr[3]  # NEW
    for key in E_true_all.keys():
        mask_nonzero = E_true_all[key] > 0
        resolution = (E_reco_all[key][mask_nonzero] - E_true_all[key][mask_nonzero]) / E_true_all[key][mask_nonzero]
        ax.hist(resolution, bins=100, histtype="step", linewidth=2, label=str(key))
    plt.legend()
    ax.set_xlabel(r"$(E_{reco} - E_{true}) / E_{true}$")
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    ax.set_title("Energy resolution (per patch)")

    for ax in axarr.flatten():
        if is_axes_empty(ax):
            ax.set_visible(False)

    fig.tight_layout()
    plt.show()

    if saveas is not None:
        fig.savefig(saveas+"_multi_event_figures.png")

    # -----------------------------
    # SINGLE EVENT (UPDATED)
    # -----------------------------

    r_all, phi_all, z_all, x_all, y_all = [], [], [], [], []
    E_true_all, E_reco_all = [], []

    # ✅ NEW: store per-group resolution
    resolution_per_group = {}  # NEW

    for key in batch.keys():
        local_ids = batch[key]["patch_positions"][0].detach().cpu().numpy()
        flat_true = batch[key]["flat_tensor"][0].detach().cpu().numpy()
        flat_reco = patches_chunked_reco[key][0].detach().cpu().numpy()

        E_true = np.nan_to_num(flat_true.sum(axis=1))
        E_reco = np.nan_to_num(flat_reco.sum(axis=1))

        # store for global scatter
        x_all.append(local_ids[:, 0]*np.cos(local_ids[:, 1]))
        y_all.append(local_ids[:, 0]*np.sin(local_ids[:, 1]))
        z_all.append(local_ids[:, 2])
        r_all.append(local_ids[:, 0])
        phi_all.append(local_ids[:, 1])
        E_true_all.append(E_true)
        E_reco_all.append(E_reco)

        # ---------------------------------------
        # NEW: per-group resolution (NOT aggregated)
        # ---------------------------------------
        
        mask_nonzero = E_true > 0
        res = (E_reco[mask_nonzero] - E_true[mask_nonzero]) / E_true[mask_nonzero]
        resolution_per_group[str(key)] = res  # NEW

    r = np.concatenate(r_all)
    phi = np.concatenate(phi_all)
    z = np.concatenate(z_all)
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    E_true = np.concatenate(E_true_all)
    E_reco = np.concatenate(E_reco_all)

    eps = 1e-8
    rel_err = (E_reco - E_true) / (E_true + eps)

    # robust color scaling
    vmax = np.percentile(np.abs(rel_err), 99)
    vmin = -vmax

    fig, axarr = plt.subplots(1, 4, figsize=(7*4, 6), constrained_layout=True)  # unchanged

    def scatter_plot(ax, x, y, c, title, vmin, vmax):
        sc = ax.scatter(
            x, y,
            c=c,
            s=10,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(title)
        #ax.set_aspect('equal', adjustable='box')  # ✅ FIX
        return sc


    # -------------------------
    # r-phi
    # -------------------------
    sc0 = scatter_plot(axarr[0], x, y, rel_err, " (Reco - True)/True : r-phi", vmin, vmax)
    axarr[0].set_xlabel("x")
    axarr[0].set_ylabel("y")

    # -------------------------
    # z-phi
    # -------------------------
    sc1 = scatter_plot(axarr[1], z, phi, rel_err, " (Reco - True)/True : z-phi", vmin, vmax)
    axarr[1].set_xlabel("z")
    axarr[1].set_ylabel("phi")

    # -------------------------
    # r-z
    # -------------------------
    sc2 = scatter_plot(axarr[2], r, z, rel_err, " (Reco - True)/True : r-z", vmin, vmax)
    axarr[2].set_xlabel("r")
    axarr[2].set_ylabel("z")

    fig.colorbar(sc0, ax=axarr[0])
    fig.colorbar(sc1, ax=axarr[1])
    fig.colorbar(sc2, ax=axarr[2])


    ax = axarr[3] # NEW 
    for key, res in resolution_per_group.items(): 
        ax.hist(res, bins=50, histtype="step", linewidth=1.5, label=str(key)) # NEW 
        ax.set_xlabel(r"$(E_{reco} - E_{true}) / E_{true}$") 
        ax.set_ylabel("Counts") 
        ax.set_yscale("log") 
        ax.set_title("Resolution per patch group") 
        ax.legend(fontsize=6) # NEW


    plt.show()

    if saveas is not None:
        fig.savefig(saveas + "_single_event_figures.png")
    plt.close()



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