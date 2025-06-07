import numpy as np
import math
import awkward
import vector

from scipy.sparse import coo_matrix

# Constants
B = -2.0  # magnetic field in T (-2 for CLD FCC-ee)


def get_sitrack_links(ev):
    return ev.arrays(
        [
            "SiTracksMCTruthLink.weight",
            "_SiTracksMCTruthLink_to/_SiTracksMCTruthLink_to.collectionID",
            "_SiTracksMCTruthLink_to/_SiTracksMCTruthLink_to.index",
            "_SiTracksMCTruthLink_from/_SiTracksMCTruthLink_from.collectionID",
            "_SiTracksMCTruthLink_from/_SiTracksMCTruthLink_from.index",
        ]
    )


def get_tracker_hits_begin_end(ev):
    return ev.arrays(
        [
            "SiTracks_Refitted/SiTracks_Refitted.trackerHits_begin",
            "SiTracks_Refitted/SiTracks_Refitted.trackerHits_end",
            "_SiTracks_Refitted_trackerHits/_SiTracks_Refitted_trackerHits.index",
            "_SiTracks_Refitted_trackerHits/_SiTracks_Refitted_trackerHits.collectionID",
        ]
    )


def get_cluster_hits_begin_end(ev):
    return ev.arrays(
        [
            "PandoraClusters/PandoraClusters.hits_begin",
            "PandoraClusters/PandoraClusters.hits_end",
            "_PandoraClusters_hits/_PandoraClusters_hits.index",
            "_PandoraClusters_hits/_PandoraClusters_hits.collectionID",
        ]
    )


def get_calohit_links(ev):
    return ev.arrays(
        [
            "CalohitMCTruthLink.weight",
            "_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.collectionID",
            "_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.index",
            "_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.collectionID",
            "_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.index",
        ]
    )


def get_calo_hit_data(ev):
    return ev.arrays(
        [
            "ECALBarrel",
            "ECALEndcap",
            "HCALBarrel",
            "HCALEndcap",
            "HCALOther",
            "MUON",
        ]
    )


def get_tracker_hit_data(ev):
    return ev.arrays(
        [
            "VXDTrackerHits",
            "VXDEndcapTrackerHits",
            "ITrackerHits",
            "OTrackerHits",
            # These need to be added to the keep statements of the next generation
            # "ITrackerEndcapHits",
            # "OTrackerEndcapHits",
        ]
    )


def get_track_data(ev):
    track_data = ev.arrays(
        [
            "SiTracks_Refitted",
            "SiTracks_Refitted_dQdx",
            "_SiTracks_Refitted_trackStates",
        ]
    )

    return track_data


def get_cluster_data(ev):
    cluster_data = ev.arrays(
        [
            "PandoraClusters",
            "_PandoraClusters_hits",
        ]
    )

    return cluster_data


def get_gen_data(ev):
    gen_data = ev.arrays(["MCParticles"])
    return gen_data


def get_event_data(ev):
    """
    Retrieves all data entries returned by the existing functions using ev.arrays().

    Args:
        ev: The event data object.

    Returns:
        dict: A dictionary containing all data entries.
    """
    return ev.arrays(
        [
            # SiTrack links
            "SiTracksMCTruthLink.weight",
            "_SiTracksMCTruthLink_to/_SiTracksMCTruthLink_to.collectionID",
            "_SiTracksMCTruthLink_to/_SiTracksMCTruthLink_to.index",
            "_SiTracksMCTruthLink_from/_SiTracksMCTruthLink_from.collectionID",
            "_SiTracksMCTruthLink_from/_SiTracksMCTruthLink_from.index",
            # Tracker hits begin/end
            "SiTracks_Refitted/SiTracks_Refitted.trackerHits_begin",
            "SiTracks_Refitted/SiTracks_Refitted.trackerHits_end",
            "_SiTracks_Refitted_trackerHits/_SiTracks_Refitted_trackerHits.index",
            "_SiTracks_Refitted_trackerHits/_SiTracks_Refitted_trackerHits.collectionID",
            # Cluster hits begin/end
            "PandoraClusters/PandoraClusters.hits_begin",
            "PandoraClusters/PandoraClusters.hits_end",
            "_PandoraClusters_hits/_PandoraClusters_hits.index",
            "_PandoraClusters_hits/_PandoraClusters_hits.collectionID",
            # Calo hit links
            "CalohitMCTruthLink.weight",
            "_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.collectionID",
            "_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.index",
            "_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.collectionID",
            "_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.index",
            # Calo hit data
            "ECALBarrel",
            "ECALEndcap",
            "HCALBarrel",
            "HCALEndcap",
            "HCALOther",
            "MUON",
            # Tracker hit data
            "VXDTrackerHits",
            "VXDEndcapTrackerHits",
            "ITrackerHits",
            "OTrackerHits",
            # Track data
            "SiTracks_Refitted",
            "SiTracks_Refitted_dQdx",
            "_SiTracks_Refitted_trackStates",
            # Cluster data
            "PandoraClusters",
            "_PandoraClusters_hits",
            # Gen data
            "MCParticles",
            "MCParticles.parents_begin",
            "MCParticles.parents_end",
            "_MCParticles_parents/_MCParticles_parents.index",
            "MCParticles.daughters_begin",
            "MCParticles.daughters_end",
            "_MCParticles_daughters/_MCParticles_daughters.index",
        ]
    )


def hits_to_features(hit_data, iev, coll, feats):
    """
    Converts hit data into a structured feature array for a specific event and collection.

    Args:
        hit_data (dict): A dictionary containing hit data, where keys are strings representing
            collection and feature names, and values are arrays of feature data.
        iev (int): The index of the event to extract data for.
        coll (str): The name of the hit collection (e.g., "VXDTrackerHits", "VXDEndcapTrackerHits",
            "ECALBarrel", "ECALEndcap", etc.).
        feats (list of str): A list of feature names to extract from the hit data
            (e.g., "position.x", "position.y", "position.z", "energy", "type", etc.).

    Returns:
        awkward.Array: An Awkward Array containing the extracted features for the specified event
            and collection. The array includes an additional "subdetector" feature, which encodes
            the subdetector type:
            - 0 for ECAL
            - 1 for HCAL
            - 2 for MUON
            - 3 for other collections.
    """
    # tracker hits store eDep instead of energy
    if "TrackerHit" in coll or "TrackerEndcapHits" in coll:
        new_feats = []
        for feat in feats:
            feat_to_get = feat
            if feat == "energy":
                feat_to_get = "eDep"
            new_feats.append((feat, feat_to_get))
    else:
        new_feats = [(f, f) for f in feats]

    feat_arr = {f1: hit_data[coll + "." + f2][iev] for f1, f2 in new_feats}

    sdcoll = "subdetector"
    feat_arr[sdcoll] = np.zeros(len(feat_arr["type"]), dtype=np.int32)
    if coll.startswith("ECAL"):
        feat_arr[sdcoll][:] = 0
    elif coll.startswith("HCAL"):
        feat_arr[sdcoll][:] = 1
    elif coll.startswith("MUON"):
        feat_arr[sdcoll][:] = 2
    else:
        feat_arr[sdcoll][:] = 3
    return awkward.Array(feat_arr)


def genparticle_track_adj(event_data, iev):
    trk_to_gen_trkidx = event_data["_SiTracksMCTruthLink_from/_SiTracksMCTruthLink_from.index"][iev]
    trk_to_gen_genidx = event_data["_SiTracksMCTruthLink_to/_SiTracksMCTruthLink_to.index"][iev]
    trk_to_gen_w = event_data["SiTracksMCTruthLink.weight"][iev]

    genparticle_to_track_matrix_coo0 = awkward.to_numpy(trk_to_gen_genidx)
    genparticle_to_track_matrix_coo1 = awkward.to_numpy(trk_to_gen_trkidx)
    genparticle_to_track_matrix_w = awkward.to_numpy(trk_to_gen_w)

    return genparticle_to_track_matrix_coo0, genparticle_to_track_matrix_coo1, genparticle_to_track_matrix_w


def produce_gp_to_track(ev, iev, num_genparticles, num_tracks):
    """
    Produces the genparticle-to-track adjacency matrix (gp_to_track) in dense format.

    Args:
        iev: The event index.
        num_genparticles: Total number of genparticles.
        num_tracks: Total number of tracks.

    Returns:
        gp_to_track: A dense adjacency matrix where each entry represents the weight
                     between a genparticle and a track.
    """
    # Get the COO format adjacency data
    genparticle_to_track_matrix_coo0, genparticle_to_track_matrix_coo1, genparticle_to_track_matrix_w = (
        genparticle_track_adj(ev, iev)
    )

    # Create a sparse matrix for the association between gen particles and tracks
    if len(genparticle_to_track_matrix_coo0) > 0:
        gp_to_track = coo_matrix(
            (genparticle_to_track_matrix_w, (genparticle_to_track_matrix_coo0, genparticle_to_track_matrix_coo1)),
            shape=(num_genparticles, num_tracks),
        ).todense()
    else:
        gp_to_track = np.zeros((num_genparticles, 1))

    return gp_to_track


def create_global_to_local_mapping(hit_data, iev, collectionIDs):
    """Create a mapping from global hit indices to local (collection, index) pairs."""
    hit_idx_global_to_local = {}
    hit_idx_global = 0

    for col in sorted(hit_data.fields):
        icol = collectionIDs[col]
        for ihit in range(len(hit_data[col][col + ".position.x"][iev])):
            hit_idx_global_to_local[hit_idx_global] = (icol, ihit)
            hit_idx_global += 1

    hit_idx_local_to_global = {v: k for k, v in hit_idx_global_to_local.items()}

    return hit_idx_global_to_local, hit_idx_local_to_global


def create_hit_feature_matrix(hit_data, iev, feats):
    """Extract features from hit data and create a feature matrix."""
    hit_feature_matrix = []

    for col in sorted(hit_data.fields):
        hit_features = hits_to_features(hit_data[col], iev, col, feats)
        hit_feature_matrix.append(hit_features)

    # Combine all hit features into a single Record
    hit_feature_matrix = {
        k: awkward.concatenate([hit_feature_matrix[i][k] for i in range(len(hit_feature_matrix))])
        for k in hit_feature_matrix[0].fields
    }

    return hit_feature_matrix


# Combine the two functions above into one to reduce the number of loops over hit data.
def create_hit_feature_matrix_and_mapping(hit_data, iev, collectionIDs, feats):
    """Combines the creation of hit feature matrix and global-local index mapping in one loop over hit data."""

    # Initialize global hit index mapping
    hit_idx_global = 0
    hit_idx_global_to_local = {}
    hit_feature_matrix = []

    # Process hit data to create feature matrix and global-local mappings
    for col in sorted(hit_data.fields):
        icol = collectionIDs[col]
        hit_features = hits_to_features(hit_data[col], iev, col, feats)
        hit_feature_matrix.append(hit_features)
        for ihit in range(len(hit_data[col][col + ".position.x"][iev])):
            hit_idx_global_to_local[hit_idx_global] = (icol, ihit)
            hit_idx_global += 1
    hit_idx_local_to_global = {v: k for k, v in hit_idx_global_to_local.items()}
    hit_feature_matrix = {
        k: np.concatenate([hit_feature_matrix[i][k].to_numpy() for i in range(len(hit_feature_matrix))])
        for k in hit_feature_matrix[0].fields
    }

    return hit_feature_matrix, hit_idx_global_to_local, hit_idx_local_to_global


# TODO: check correctnes of this function
def create_track_to_hit_coo_matrix(event_data, iev, collectionIDs):
    """
    Creates the COO matrix indices and weights for the relationship between tracks and tracker hits.

    Args:
        ev: The event data containing track property data.
        iev: The index of the event to extract data for.
        collectionIDs: A dictionary mapping collection names to their IDs.

    Returns:
        tuple: A tuple containing three arrays:
            - Row indices (track indices).
            - Column indices (global hit indices).
            - Weights (association weights between tracks and hits).
    """
    # Extract tracker hit data
    tracker_hit_data = event_data[
        [
            "VXDTrackerHits",
            "VXDEndcapTrackerHits",
            "ITrackerHits",
            "OTrackerHits",
            # "ITrackerEndcapHits",
            # "OTrackerEndcapHits",
        ]
    ]

    # Extract tracker hit to track associations
    hit_beg = event_data["SiTracks_Refitted/SiTracks_Refitted.trackerHits_begin"][
        iev
    ]  # hit_beg[i] gives the first hit index for track i
    hit_end = event_data["SiTracks_Refitted/SiTracks_Refitted.trackerHits_end"][
        iev
    ]  # hit_end[i] gives the last hit index for track i
    trk_hit_idx = event_data["_SiTracks_Refitted_trackerHits/_SiTracks_Refitted_trackerHits.index"][iev]
    trk_hit_coll = event_data["_SiTracks_Refitted_trackerHits/_SiTracks_Refitted_trackerHits.collectionID"][iev]

    # Create a mapping from global hit indices to local (collection, index) pairs
    _, hit_idx_local_to_global = create_global_to_local_mapping(tracker_hit_data, iev, collectionIDs)

    # Initialize lists for COO matrix
    track_to_hit_matrix_coo0 = []
    track_to_hit_matrix_coo1 = []
    track_to_hit_matrix_w = []

    # Iterate over tracks and their associated hits
    for track_idx, (beg, end) in enumerate(zip(hit_beg, hit_end)):
        for ihit in range(beg, end):
            local_hit_idx = trk_hit_idx[ihit]
            collid = trk_hit_coll[ihit]

            if (collid, local_hit_idx) not in hit_idx_local_to_global:
                continue
            global_hit_idx = hit_idx_local_to_global[(collid, local_hit_idx)]

            # Append to COO matrix
            track_to_hit_matrix_coo0.append(track_idx)
            track_to_hit_matrix_coo1.append(global_hit_idx)
            track_to_hit_matrix_w.append(1.0)  # Assuming weight is 1.0 for all associations

    return (
        (
            np.array(track_to_hit_matrix_coo0),
            np.array(track_to_hit_matrix_coo1),
            np.array(track_to_hit_matrix_w),
        ),
        hit_idx_local_to_global,
    )


# TODO: check correctness of this function
def create_cluster_to_hit_coo_matrix(event_data, iev, collectionIDs):
    """
    Creates the COO matrix indices and weights for the relationship between clusters and calorimeter hits.

    Args:
        ev: The event data containing cluster property data.
        iev: The index of the event to extract data for.
        collectionIDs: A dictionary mapping collection names to their IDs.

    Returns:
        tuple: A tuple containing three arrays:
            - Row indices (cluster indices).
            - Column indices (global hit indices).
            - Weights (association weights between clusters and hits).
    """
    # Extract calorimeter hit data
    calo_hit_data = event_data[
        [
            "ECALBarrel",
            "ECALEndcap",
            "HCALBarrel",
            "HCALEndcap",
            "HCALOther",
            "MUON",
        ]
    ]

    # Extract cluster-to-hit associations
    cluster_hit_begin = event_data["PandoraClusters/PandoraClusters.hits_begin"][iev]
    cluster_hit_end = event_data["PandoraClusters/PandoraClusters.hits_end"][iev]
    cluster_hit_idx = event_data["_PandoraClusters_hits/_PandoraClusters_hits.index"][iev]
    cluster_hit_coll = event_data["_PandoraClusters_hits/_PandoraClusters_hits.collectionID"][iev]

    _, calo_hit_idx_local_to_global = create_global_to_local_mapping(calo_hit_data, iev, collectionIDs)

    # Initialize lists for COO matrix
    cluster_to_hit_matrix_coo0 = []
    cluster_to_hit_matrix_coo1 = []
    cluster_to_hit_matrix_w = []

    # Iterate over clusters and their associated hits
    for cluster_idx, (beg, end) in enumerate(zip(cluster_hit_begin, cluster_hit_end)):
        for ihit in range(beg, end):
            local_hit_idx = cluster_hit_idx[ihit]
            collid = cluster_hit_coll[ihit]

            if (collid, local_hit_idx) not in calo_hit_idx_local_to_global:
                continue
            global_hit_idx = calo_hit_idx_local_to_global[(collid, local_hit_idx)]

            # Append to COO matrix
            cluster_to_hit_matrix_coo0.append(cluster_idx)
            cluster_to_hit_matrix_coo1.append(global_hit_idx)
            cluster_to_hit_matrix_w.append(1.0)  # Assuming weight is 1.0 for all associations

    return (
        (
            np.array(cluster_to_hit_matrix_coo0),
            np.array(cluster_to_hit_matrix_coo1),
            np.array(cluster_to_hit_matrix_w),
        ),
        calo_hit_idx_local_to_global,
    )


def process_calo_hit_data(event_data, iev, collectionIDs):
    feats = ["type", "energy", "position.x", "position.y", "position.z"]

    calo_hit_data = event_data[
        [
            "ECALBarrel",
            "ECALEndcap",
            "HCALBarrel",
            "HCALEndcap",
            "HCALOther",
            "MUON",
        ]
    ]
    calohit_links = event_data[
        [
            "CalohitMCTruthLink.weight",
            "_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.collectionID",
            "_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.index",
            "_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.collectionID",
            "_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.index",
        ]
    ]

    # Create a mapping from global hit indices to local (collection, index) pairs and hit feature matrix
    hit_features, _, hit_idx_local_to_global = create_hit_feature_matrix_and_mapping(
        calo_hit_data, iev, collectionIDs, feats
    )

    # Add all edges from genparticle to calohit
    calohit_to_gen_weight = calohit_links["CalohitMCTruthLink.weight"][iev]
    calohit_to_gen_calo_colid = calohit_links["_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.collectionID"][iev]
    calohit_to_gen_gen_colid = calohit_links["_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.collectionID"][iev]
    calohit_to_gen_calo_idx = calohit_links["_CalohitMCTruthLink_from/_CalohitMCTruthLink_from.index"][iev]
    calohit_to_gen_gen_idx = calohit_links["_CalohitMCTruthLink_to/_CalohitMCTruthLink_to.index"][iev]

    genparticle_to_hit_matrix_coo0 = []
    genparticle_to_hit_matrix_coo1 = []
    genparticle_to_hit_matrix_w = []
    for calo_colid, calo_idx, gen_colid, gen_idx, weight in zip(
        calohit_to_gen_calo_colid,
        calohit_to_gen_calo_idx,
        calohit_to_gen_gen_colid,
        calohit_to_gen_gen_idx,
        calohit_to_gen_weight,
    ):
        genparticle_to_hit_matrix_coo0.append(gen_idx)
        genparticle_to_hit_matrix_coo1.append(hit_idx_local_to_global[(calo_colid, calo_idx)])
        genparticle_to_hit_matrix_w.append(weight)

    return (
        hit_features,
        (
            np.array(genparticle_to_hit_matrix_coo0),
            np.array(genparticle_to_hit_matrix_coo1),
            np.array(genparticle_to_hit_matrix_w),
        ),
        hit_idx_local_to_global,
    )


def process_tracker_hit_data(event_data, iev, collectionIDs):

    feats = ["type", "energy", "position.x", "position.y", "position.z"]

    tracker_hit_data = event_data[
        [
            "VXDTrackerHits",
            "VXDEndcapTrackerHits",
            "ITrackerHits",
            "OTrackerHits",
            # "ITrackerEndcapHits",
            # "OTrackerEndcapHits",
        ]
    ]

    # Create a mapping from global hit indices to local (collection, index) pairs and hit feature matrix
    hit_feature_matrix, _, hit_idx_local_to_global = create_hit_feature_matrix_and_mapping(
        tracker_hit_data, iev, collectionIDs, feats
    )

    # Extract tracker hit to track associations
    hit_beg = event_data["SiTracks_Refitted/SiTracks_Refitted.trackerHits_begin"][
        iev
    ]  # hit_beg[i] gives the first hit index for track i
    hit_end = event_data["SiTracks_Refitted/SiTracks_Refitted.trackerHits_end"][
        iev
    ]  # hit_end[i] gives the last hit index for track i
    trk_hit_idx = event_data["_SiTracks_Refitted_trackerHits/_SiTracks_Refitted_trackerHits.index"][iev]
    trk_hit_coll = event_data["_SiTracks_Refitted_trackerHits/_SiTracks_Refitted_trackerHits.collectionID"][iev]

    # Get the COO format adjacency data
    genparticle_to_track_matrix_coo0, genparticle_to_track_matrix_coo1, genparticle_to_track_matrix_w = (
        genparticle_track_adj(event_data, iev)
    )
    gen_indices = genparticle_to_track_matrix_coo0
    track_indices = genparticle_to_track_matrix_coo1

    # Initialize lists for COO matrix
    genparticle_to_hit_matrix_coo0 = []
    genparticle_to_hit_matrix_coo1 = []
    genparticle_to_hit_matrix_w = []

    # Iterate over non-zero elements to find links between genparticles and tracks
    for gen_idx, track_idx, weight in zip(gen_indices, track_indices, genparticle_to_track_matrix_w):
        if weight > 0:  # Only consider non-zero weights
            # Find tracker hits associated with the track
            for ihit in range(hit_beg[track_idx], hit_end[track_idx]):  # for all hits in this track
                # Translate local hit index to global hit index
                local_hit_idx = trk_hit_idx[ihit]
                collid = trk_hit_coll[ihit]

                if (
                    collid,
                    local_hit_idx,
                ) not in hit_idx_local_to_global:  # Check if the hit is in the local-to-global mapping
                    continue
                global_hit_idx = hit_idx_local_to_global[(collid, local_hit_idx)]

                # Append the gp to hit association and weight to the COO matrix
                genparticle_to_hit_matrix_coo0.append(gen_idx)
                genparticle_to_hit_matrix_coo1.append(global_hit_idx)
                genparticle_to_hit_matrix_w.append(weight)

    return (
        hit_feature_matrix,  # Tracker hit feature matrix
        (
            np.array(genparticle_to_hit_matrix_coo0),
            np.array(genparticle_to_hit_matrix_coo1),
            np.array(genparticle_to_hit_matrix_w),
        ),
        hit_idx_local_to_global,
    )


def gen_to_features(event_data, iev):

    gen_data = event_data["MCParticles"]

    mc_coll = "MCParticles"
    gen_arr = gen_data[iev]

    gen_arr = {k.replace(mc_coll + ".", ""): gen_arr[k] for k in gen_arr.fields}

    MCParticles_p4 = vector.awk(
        awkward.zip(
            {
                "mass": gen_arr["mass"],
                "x": gen_arr["momentum.x"],
                "y": gen_arr["momentum.y"],
                "z": gen_arr["momentum.z"],
            }
        )
    )
    gen_arr["pt"] = MCParticles_p4.pt
    gen_arr["eta"] = MCParticles_p4.eta
    gen_arr["phi"] = MCParticles_p4.phi
    gen_arr["energy"] = MCParticles_p4.energy
    gen_arr["sin_phi"] = np.sin(gen_arr["phi"])
    gen_arr["cos_phi"] = np.cos(gen_arr["phi"])

    ret = {
        "PDG": gen_arr["PDG"],
        "generatorStatus": gen_arr["generatorStatus"],
        "charge": gen_arr["charge"],
        "pt": gen_arr["pt"],
        "eta": gen_arr["eta"],
        "phi": gen_arr["phi"],
        "sin_phi": gen_arr["sin_phi"],
        "cos_phi": gen_arr["cos_phi"],
        "energy": gen_arr["energy"],
        # "ispu": gen_arr["ispu"],
        "simulatorStatus": gen_arr["simulatorStatus"],
        # "gp_to_track": np.zeros(len(gen_arr["PDG"]), dtype=np.float64),
        # "gp_to_cluster": np.zeros(len(gen_arr["PDG"]), dtype=np.float64),
        # "jet_idx": np.zeros(len(gen_arr["PDG"]), dtype=np.int64),
        # "daughters_begin": gen_arr["daughters_begin"],
        # "daughters_end": gen_arr["daughters_end"],
        "px": gen_arr["momentum.x"],
        "py": gen_arr["momentum.y"],
        "pz": gen_arr["momentum.z"],
        "mass": gen_arr["mass"],
    }

    # ret["index"] = prop_data["_MCParticles_daughters/_MCParticles_daughters.index"][iev]

    # make all values numpy arrays
    ret = {k: awkward.to_numpy(v) for k, v in ret.items()}

    return ret


# From https://bib-pubdb1.desy.de/record/81214/files/LC-DET-2006-004[1].pdf, eq12
# pT​(in GeV/c) ≈ a [mm/s] * |Bz(in T) / omega(1/mm)|
# a = c * 10^(-15) = 3*10^(-4)
def track_pt(omega, bfield=B):
    a = 3 * 10**-4
    return a * np.abs(bfield / omega)


def track_to_features(event_data, iev):
    """
    Extracts track features from the provided property data for a specific event and track collection.

    Args:
        ev (uproot.models.TTree.Model_TTree_v20): The event data containing track property data.
        iev (int): The index of the event to extract data for.

    Returns:
        awkward.Record: A record containing the extracted track features, including:
            - "type", "chi2", "ndf": Basic track properties.
            - "dEdx", "dEdxError": Energy deposition and its error.
            - "radiusOfInnermostHit": Radius of the innermost hit for each track.
            - "tanLambda", "D0", "phi", "omega", "Z0", "time": Track state properties.
            - "pt", "px", "py", "pz", "p": Momentum components and magnitude.
            - "eta": Pseudorapidity.
            - "sin_phi", "cos_phi": Sine and cosine of the azimuthal angle.
            - "elemtype": Element type (always 1 for tracks).
            - "q": Charge of the track (+1 or -1).

    Notes:
        - The function calculates additional derived features such as momentum components, pseudorapidity,
          and radius of the innermost hit.
        - The "AtFirstHit" state is used to determine the innermost hit radius.
        - The charge is set to +1 or -1 based on the sign of the "omega" parameter.
        - The input `ev` is expected to be an uproot TTree object containing the necessary branches.
    """
    track_coll = "SiTracks_Refitted"
    track_arr = event_data[track_coll][iev]
    track_arr_dQdx = event_data["SiTracks_Refitted_dQdx"][iev]
    track_arr_trackStates = event_data["_SiTracks_Refitted_trackStates"][iev]

    feats_from_track = ["type", "chi2", "ndf"]
    ret = {feat: track_arr[track_coll + "." + feat] for feat in feats_from_track}

    ret["dEdx"] = track_arr_dQdx["SiTracks_Refitted_dQdx.dQdx.value"]
    ret["dEdxError"] = track_arr_dQdx["SiTracks_Refitted_dQdx.dQdx.error"]

    # build the radiusOfInnermostHit variable
    num_tracks = len(ret["dEdx"])
    innermost_radius = []
    for itrack in range(num_tracks):

        # select the track states corresponding to itrack
        # pick the state AtFirstHit
        # https://github.com/key4hep/EDM4hep/blob/fe5a54046a91a7e648d0b588960db7841aebc670/edm4hep.yaml#L220
        ibegin = track_arr[track_coll + "." + "trackStates_begin"][itrack]
        iend = track_arr[track_coll + "." + "trackStates_end"][itrack]

        refX = track_arr_trackStates["_SiTracks_Refitted_trackStates" + "." + "referencePoint.x"][ibegin:iend]
        refY = track_arr_trackStates["_SiTracks_Refitted_trackStates" + "." + "referencePoint.y"][ibegin:iend]
        location = track_arr_trackStates["_SiTracks_Refitted_trackStates" + "." + "location"][ibegin:iend]

        istate = np.argmax(location == 2)  # 2 refers to AtFirstHit

        innermost_radius.append(math.sqrt(refX[istate] ** 2 + refY[istate] ** 2))

    ret["radiusOfInnermostHit"] = np.array(innermost_radius)

    # get the index of the first track state
    trackstate_idx = event_data[track_coll][track_coll + ".trackStates_begin"][iev]
    # get the properties of the track at the first track state (at the origin)
    for k in ["tanLambda", "D0", "phi", "omega", "Z0", "time"]:
        ret[k] = awkward.to_numpy(
            event_data["_SiTracks_Refitted_trackStates"]["_SiTracks_Refitted_trackStates." + k][iev][trackstate_idx]
        )

    ret["pt"] = awkward.to_numpy(track_pt(ret["omega"]))
    ret["px"] = awkward.to_numpy(np.cos(ret["phi"])) * ret["pt"]
    ret["py"] = awkward.to_numpy(np.sin(ret["phi"])) * ret["pt"]
    ret["pz"] = ret["pt"] * ret["tanLambda"]
    ret["p"] = np.sqrt(ret["px"] ** 2 + ret["py"] ** 2 + ret["pz"] ** 2)
    cos_theta = np.divide(ret["pz"], ret["p"], where=ret["p"] > 0)
    theta = np.arccos(cos_theta)
    tt = np.tan(theta / 2.0)
    eta = awkward.to_numpy(-np.log(tt, where=tt > 0))
    eta[tt <= 0] = 0.0
    ret["eta"] = eta

    ret["sin_phi"] = np.sin(ret["phi"])
    ret["cos_phi"] = np.cos(ret["phi"])

    # track is always type 1
    ret["elemtype"] = 1 * np.ones(num_tracks, dtype=np.float32)

    ret["q"] = ret["omega"].copy()
    ret["q"][ret["q"] > 0] = 1
    ret["q"][ret["q"] < 0] = -1

    return ret


def cluster_to_features(event_data, iev, cluster_features=["position.x", "position.y", "position.z", "energy"]):
    """
    Extracts cluster features for a specific event.

    Args:
        ev: The event data containing cluster property data.
        iev: The index of the event to extract data for.
        cluster_features (list of str): A list of cluster feature names to extract.
            Default is ["position.x", "position.y", "position.z", "energy"].
    Returns:
        dict: A dictionary containing cluster features for the specified event.
    Raises:
        ValueError: If a specified feature is not found in the event data.
    """
    cluster_data = event_data["PandoraClusters"]
    for feat in cluster_features:
        if f"PandoraClusters.{feat}" not in cluster_data.fields:
            raise ValueError(f"Feature {feat} not found in PandoraClusters.")
        # Extract cluster features

    return {f"{feat}": awkward.to_numpy(cluster_data[f"PandoraClusters.{feat}"][iev]) for feat in cluster_features}


def create_genparticle_to_genparticle_coo_matrix(event_data, iev):
    """
    Creates the COO matrix indices and weights for the relationship between genparticles
    based on parent-daughter associations.

    Args:
        ev: The event data containing genparticle property data.
        iev: The index of the event to extract data for.

    Returns:
        tuple: A tuple containing three arrays:
            - Row indices (parent genparticle indices).
            - Column indices (daughter genparticle indices).
            - Weights (association weights between parent and daughter genparticles, set to 1.0).
    """
    # Extract parent-daughter associations
    daughters_begin = event_data["MCParticles.daughters_begin"][iev]
    daughters_end = event_data["MCParticles.daughters_end"][iev]
    daughter_indices = event_data["_MCParticles_daughters/_MCParticles_daughters.index"][iev]

    # Initialize lists for COO matrix
    coo_rows = []
    coo_cols = []
    coo_weights = []

    # Iterate over genparticles and their associated daughters
    for parent_idx, (beg, end) in enumerate(zip(daughters_begin, daughters_end)):
        for idaughter in range(beg, end):
            daughter_idx = daughter_indices[idaughter]
            coo_rows.append(parent_idx)
            coo_cols.append(daughter_idx)
            coo_weights.append(1.0)  # Assuming weight is 1.0 for all associations

    return (
        np.array(coo_rows),
        np.array(coo_cols),
        np.array(coo_weights),
    )


def create_genparticle_to_genparticle_coo_matrix2(event_data, iev):
    """
    Creates the COO matrix indices and weights for the relationship between genparticles
    based on parent-daughter associations.

    Args:
        ev: The event data containing genparticle property data.
        iev: The index of the event to extract data for.

    Returns:
        tuple: A tuple containing three arrays:
            - Row indices (parent genparticle indices).
            - Column indices (daughter genparticle indices).
            - Weights (association weights between parent and daughter genparticles, set to 1.0).
    """
    parents_begin = event_data["MCParticles.parents_begin"][iev]
    parents_end = event_data["MCParticles.parents_end"][iev]
    parent_indices = event_data["_MCParticles_parents/_MCParticles_parents.index"][iev]

    # Initialize lists for COO matrix
    coo_rows = []
    coo_cols = []
    coo_weights = []

    # Iterate over genparticles and their associated parents
    for daughter_idx, (beg, end) in enumerate(zip(parents_begin, parents_end)):
        for iparent in range(beg, end):
            parent_idx = parent_indices[iparent]
            coo_rows.append(parent_idx)
            coo_cols.append(daughter_idx)
            coo_weights.append(1.0)  # Assuming weight is 1.0 for all associations

    return (
        np.array(coo_rows),
        np.array(coo_cols),
        np.array(coo_weights),
    )
