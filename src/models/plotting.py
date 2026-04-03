
def plot_model(batch, patches_chunked_reco, vq_out, num_codes, device="cuda", vit_kwargs={}, n_scatterpoints_to_plot=300, saveas=None):

    def is_axes_empty(ax):
        return not (ax.lines or ax.patches or ax.collections or ax.images or ax.texts or ax.artists or ax.tables)

    # -----------------------------
    # LATENT + CODEBOOK (UNCHANGED)
    # -----------------------------
    master_z_q = vq_out["z_q"].squeeze(2).detach().cpu().numpy() # (B, P, LATENT_DIM)
    master_z_e = vq_out["z"].squeeze(2).detach().cpu().numpy() # (B, P, LATENT_DIM)
    master_idx = vq_out["q"].squeeze(2).detach().cpu().numpy() # (B, P) 



    # flatten across all batches
    z_q_concat = np.concatenate([master_z_q[i] for i in range(len(master_z_q))]) # (B*P, LATENT_DIM)
    z_e_concat = np.concatenate([master_z_e[i] for i in range(len(master_z_e))]) # (B*P, LATENT_DIM)
    idx_concat = np.concatenate([master_idx[i] for i in range(len(master_idx))]) # (B*P)


    # ✅ CHANGED: now 4 panels (added resolution)
    fig, axarr = plt.subplots(1, 4, figsize=(7*4, 7))  # CHANGED

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


