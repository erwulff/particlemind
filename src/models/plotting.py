
def plot_model_hits(model, input_data, labels, device="cuda", n_events_to_plot=2, n_scatterpoints_to_plot=200, masks=None, saveas=None):
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

    # make empty axes invisible
    def is_axes_empty(ax):
        return not (ax.lines or ax.patches or ax.collections or ax.images or ax.texts or ax.artists or ax.tables)


    input_data = input_data.to(device)
    model = model.to(device)
   

    # run the model on the input data
    with torch.no_grad():
        # print(f"Model device: {next(model.parameters()).device}")
        # print(f"Samples device: {samples.device}")
        reco, vq_out = model(input_data, masks)
        
        master_z_q = vq_out["z_q"]
        master_z_e = vq_out["z"]
        master_idx = vq_out["q"]

        # move r, z_e, z_q, idx to cpu for plotting
        reco = reco.detach().cpu().numpy()
        master_z_e = master_z_e.detach().cpu().numpy()
        master_z_q = master_z_q.detach().cpu().numpy()
        master_idx = master_idx.detach().cpu().numpy()

    input_data = input_data.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    if masks is not None:
        masks = masks.detach().cpu().numpy()

    event_samples_E, event_samples_x, event_samples_y,event_samples_z = [], [], [], []
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
    bins = np.linspace(np.min(event_samples_E_concat), np.max(event_samples_E_concat), 50)
    ax.hist(event_samples_E_concat, bins=bins, label="samples", density=True, histtype="step", linewidth=2)
    ax.hist(reco_samples_E_concat, bins=bins, label="reco", density=True, histtype="step", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("$E$")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")

    # histogram the difference in energy
    ax = axarr[1]
    ax.hist((event_samples_E_concat - reco_samples_E_concat)/event_samples_E_concat, bins=50, density=True, histtype="step", linewidth=2)
    ax.set_xlabel("$E_{true} - E_{reco}$ /$E_{true}$ ")
    ax.set_ylabel("Density")
    ax.set_yscale("log")

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

    ax.hist((np.array(hit_clusters_true) - np.array(hit_clusters_reco))/np.array(hit_clusters_true), bins=50, density=True, histtype="step", linewidth=2)
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

    bins_x = np.linspace(np.min(single_event_samples_x), np.max(single_event_samples_x), 100)
    bins_y = np.linspace(np.min(single_event_samples_y), np.max(single_event_samples_y), 100)
    bins_z = np.linspace(np.min(single_event_samples_z), np.max(single_event_samples_z), 100)

    fig, axarr = plt.subplots(1, 6, figsize=(7*6, 6))

    # data, x-y
    ax = axarr[0]
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


