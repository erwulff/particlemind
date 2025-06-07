import uproot
import h5py
import awkward
import argparse
from tqdm import tqdm

from pathlib import Path

from cld_processing import (
    get_event_data,
    gen_to_features,
    track_to_features,
    cluster_to_features,
    process_calo_hit_data,
    process_tracker_hit_data,
    create_track_to_hit_coo_matrix,
    create_cluster_to_hit_coo_matrix,
    genparticle_track_adj,
    create_genparticle_to_genparticle_coo_matrix,
)


def process_root_files(input_dir, output_dir, events_per_hdf5=None, max_events=None, max_root_files=None):
    """
    This function reads ROOT files from the specified input directory, extracts event data,
    and saves the features and adjacency matrices into HDF5 files in the specified output directory.
    It supports processing a limited number of events or ROOT files, and can create one HDF5 file
    per ROOT file or group events into multiple HDF5 files.

    Args:
        input_dir (str or Path): Path to the directory containing the ROOT files to process.
        output_dir (str or Path): Path to the directory where the HDF5 files will be saved.
        events_per_hdf5 (int, optional): Number of events to store in each HDF5 file. If None or <= 0,
            one HDF5 file will be created per ROOT file. Defaults to None.
        max_events (int, optional): Maximum number of events to process across all files. If None,
            all events will be processed. Defaults to None.
        max_root_files (int, optional): Maximum number of ROOT files to process. If None, all ROOT
            files in the input directory will be processed. Defaults to None.
    Returns:
        None: The function saves the processed data into HDF5 files in the specified output directory.
    """
    event_counter = 0
    root_counter = 0

    # output_file = output_dir / Path(f"events_{event_counter}_to_{event_counter + events_per_hdf5 - 1}.hdf5")
    # h5f = h5py.File(output_file, "w")
    # print(f"Created new HDF5 file: {output_file}")

    one_file_per_root = events_per_hdf5 is None or events_per_hdf5 <= 0

    if len(list(Path(input_dir).rglob("*.root"))) == 0:
        raise FileNotFoundError(
            f"No ROOT files found in the specified input directory: {input_dir}. Please check the path."
        )

    for root_file in tqdm(
        Path(input_dir).rglob("*.root"),
        desc="Processing ROOT files",
        total=max_root_files or len(list(Path(input_dir).rglob("*.root"))),
    ):
        try:
            fi = uproot.open(root_file)
            collectionIDs = {
                k: v
                for k, v in zip(
                    fi.get("podio_metadata").arrays("events___idTable/m_names")["events___idTable/m_names"][0],
                    fi.get("podio_metadata").arrays("events___idTable/m_collectionIDs")[
                        "events___idTable/m_collectionIDs"
                    ][0],
                )
            }
            ev = fi["events"]
            event_data = get_event_data(ev)

            if one_file_per_root:
                output_file = output_dir / Path(f"{root_file.stem}.hdf5")
                h5f = h5py.File(output_file, "w")
                print(f"Created new HDF5 file: {output_file}")
            else:
                output_file = output_dir / Path(f"events_{event_counter}_to_{event_counter + events_per_hdf5 - 1}.hdf5")
                h5f = h5py.File(output_file, "w")
                print(f"Created new HDF5 file: {output_file}")

            for iev in range(len(ev["MCParticles.momentum.x"].array())):
                # Create a group for the event
                event_group = h5f.create_group(f"event_{event_counter}")

                # Extract features and adjacency matrices for the current event
                gen_features = gen_to_features(event_data, iev)
                track_features = track_to_features(event_data, iev)
                cluster_features = cluster_to_features(
                    event_data, iev, cluster_features=["position.x", "position.y", "position.z", "energy", "type"]
                )
                calo_hit_features, genparticle_to_calo_hit_matrix, _ = process_calo_hit_data(
                    event_data, iev, collectionIDs
                )
                tracker_hit_features, genparticle_to_tracker_hit_matrix, _ = process_tracker_hit_data(
                    event_data, iev, collectionIDs
                )
                track_to_tracker_hit_matrix, _ = create_track_to_hit_coo_matrix(event_data, iev, collectionIDs)
                cluster_to_cluster_hit_matrix, _ = create_cluster_to_hit_coo_matrix(event_data, iev, collectionIDs)
                gp_to_track_matrix = genparticle_track_adj(event_data, iev)
                gp_to_gp = create_genparticle_to_genparticle_coo_matrix(event_data, iev)

                # Save genparticle features
                for key, value in gen_features.items():
                    event_group.create_dataset(f"gen_features/{key}", data=value)

                # Save track features
                for key, value in track_features.items():
                    event_group.create_dataset(f"track_features/{key}", data=value)

                # Save cluster features
                for key, value in cluster_features.items():
                    event_group.create_dataset(f"cluster_features/{key}", data=value)

                # Save calorimeter hit features
                for key, value in calo_hit_features.items():
                    event_group.create_dataset(f"calo_hit_features/{key}", data=awkward.to_numpy(value))

                # Save tracker hit features
                for key, value in tracker_hit_features.items():
                    event_group.create_dataset(f"tracker_hit_features/{key}", data=awkward.to_numpy(value))

                # Save adjacency matrices
                event_group.create_dataset(
                    "genparticle_to_calo_hit_matrix/rows", data=genparticle_to_calo_hit_matrix[0]
                )
                event_group.create_dataset(
                    "genparticle_to_calo_hit_matrix/cols", data=genparticle_to_calo_hit_matrix[1]
                )
                event_group.create_dataset(
                    "genparticle_to_calo_hit_matrix/weights", data=genparticle_to_calo_hit_matrix[2]
                )

                event_group.create_dataset(
                    "genparticle_to_tracker_hit_matrix/rows", data=genparticle_to_tracker_hit_matrix[0]
                )
                event_group.create_dataset(
                    "genparticle_to_tracker_hit_matrix/cols", data=genparticle_to_tracker_hit_matrix[1]
                )
                event_group.create_dataset(
                    "genparticle_to_tracker_hit_matrix/weights", data=genparticle_to_tracker_hit_matrix[2]
                )

                event_group.create_dataset("track_to_tracker_hit_matrix/rows", data=track_to_tracker_hit_matrix[0])
                event_group.create_dataset("track_to_tracker_hit_matrix/cols", data=track_to_tracker_hit_matrix[1])
                event_group.create_dataset("track_to_tracker_hit_matrix/weights", data=track_to_tracker_hit_matrix[2])

                event_group.create_dataset("cluster_to_cluster_hit_matrix/rows", data=cluster_to_cluster_hit_matrix[0])
                event_group.create_dataset("cluster_to_cluster_hit_matrix/cols", data=cluster_to_cluster_hit_matrix[1])
                event_group.create_dataset(
                    "cluster_to_cluster_hit_matrix/weights", data=cluster_to_cluster_hit_matrix[2]
                )

                event_group.create_dataset("gp_to_track_matrix/rows", data=gp_to_track_matrix[0])
                event_group.create_dataset("gp_to_track_matrix/cols", data=gp_to_track_matrix[1])
                event_group.create_dataset("gp_to_track_matrix/weights", data=gp_to_track_matrix[2])

                event_group.create_dataset("gp_to_gp/rows", data=gp_to_gp[0])
                event_group.create_dataset("gp_to_gp/cols", data=gp_to_gp[1])
                event_group.create_dataset("gp_to_gp/weights", data=gp_to_gp[2])

                event_counter += 1

                if max_events is not None and event_counter >= max_events:
                    print(f"Reached max_events limit: {max_events}. Stopping processing.")
                    if h5f:
                        h5f.close()
                    return

                if not one_file_per_root and event_counter % events_per_hdf5 == 0:
                    if h5f:
                        h5f.close()
                    output_file = output_dir / Path(
                        f"events_{event_counter}_to_{event_counter + events_per_hdf5 - 1}.hdf5"
                    )
                    h5f = h5py.File(output_file, "w")
                    print(f"Created new HDF5 file: {output_file}")

            print(f"Processed: {root_file}")
            root_counter += 1

            if max_root_files is not None and root_counter >= max_root_files:
                print(f"Reached max_root_files limit: {max_root_files}. Stopping processing.")
                if h5f:
                    h5f.close()
                return

            # Close the HDF5 file for the current ROOT file if the flag is set
            if one_file_per_root and h5f:
                h5f.close()

        except Exception as e:
            print(f"Error processing {root_file}: {e}")
            break

    # Close the last HDF5 file if one_file_per_root is False
    if h5f and not one_file_per_root:
        h5f.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process ROOT files and save features/matrices to HDF5 format.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing ROOT files.")
    parser.add_argument("--max_events", type=int, default=None, help="Maximum number of events to process.")
    parser.add_argument("--events_per_hdf5", type=int, default=None, help="Number of events per HDF5 file.")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="Directory to save HDF5 files.")
    parser.add_argument("--max_root_files", type=int, default=None, help="Maximum number of ROOT files to process.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call the main processing function with parsed arguments
    process_root_files(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        max_events=args.max_events,
        events_per_hdf5=args.events_per_hdf5,
        max_root_files=args.max_root_files,
    )

    # load one example file and print contents
    example_file = next(output_dir.glob("*.hdf5"))
    with h5py.File(example_file, "r") as f:
        print(f"Contents of {example_file}:")
        for key in f.keys():
            print(f"- {key}: {list(f[key].keys()) if isinstance(f[key], h5py.Group) else f[key].shape}")
