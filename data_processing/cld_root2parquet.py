import uproot
import awkward as ak
import argparse
import logging
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def process_root_files_to_parquet(input_dir, output_dir, max_root_files=None):
    """
    Process ROOT files and save extracted features and matrices into Parquet files using ak arrays.

    Args:
        input_dir (str or Path): Directory containing ROOT files.
        output_dir (str or Path): Directory to save Parquet files.
        max_root_files (int, optional): Maximum number of ROOT files to process. Defaults to None.

    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_counter = 0
    total_files_to_porcess = max_root_files or len(list(Path(input_dir).rglob("*.root")))

    for root_file in tqdm(Path(input_dir).rglob("*.root"), desc="Processing ROOT files", total=total_files_to_porcess):
        if max_root_files is not None and root_counter >= max_root_files:
            logging.info(f"Reached max_root_files limit: {max_root_files}. Stopping processing.")
            break
        try:
            output_file = output_dir / f"{root_file.stem}.parquet"
            if output_file.exists():
                logging.info(f"Output file {output_file} already exists. Skipping processing.")
                root_counter += 1
                continue

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

            # Combine all events in the current ROOT file
            combined_data_dict = {
                "gen_features": [],
                "track_features": [],
                "cluster_features": [],
                "calo_hit_features": [],
                "tracker_hit_features": [],
                "genparticle_to_calo_hit_matrix": [],
                "genparticle_to_tracker_hit_matrix": [],
                "track_to_tracker_hit_matrix": [],
                "cluster_to_cluster_hit_matrix": [],
                "gp_to_track_matrix": [],
                "gp_to_gp": [],
            }

            for iev in tqdm(
                range(len(ev["MCParticles.momentum.x"].array())),
                total=len(ev["MCParticles.momentum.x"].array()),
                desc=f"Processing events in {root_file}",
                leave=False,
            ):
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

                # Append the event data to the combined data dictionary
                combined_data_dict["gen_features"].append(gen_features)
                combined_data_dict["track_features"].append(track_features)
                combined_data_dict["cluster_features"].append(cluster_features)
                combined_data_dict["calo_hit_features"].append(calo_hit_features)
                combined_data_dict["tracker_hit_features"].append(tracker_hit_features)
                combined_data_dict["genparticle_to_calo_hit_matrix"].append(genparticle_to_calo_hit_matrix)
                combined_data_dict["genparticle_to_tracker_hit_matrix"].append(genparticle_to_tracker_hit_matrix)
                combined_data_dict["track_to_tracker_hit_matrix"].append(track_to_tracker_hit_matrix)
                combined_data_dict["cluster_to_cluster_hit_matrix"].append(cluster_to_cluster_hit_matrix)
                combined_data_dict["gp_to_track_matrix"].append(gp_to_track_matrix)
                combined_data_dict["gp_to_gp"].append(gp_to_gp)

            # Convert lists to ak arrays
            for key in tqdm(
                combined_data_dict.keys(), total=len(combined_data_dict), desc="Converting to ak arrays", leave=False
            ):
                combined_data_dict[key] = ak.Array(combined_data_dict[key])

            # Save the combined data into a single Parquet file
            ak.to_parquet(combined_data_dict, output_file)

            root_counter += 1

        except Exception as e:
            logging.error(f"Error processing {root_file}: {e}")

    logging.info(f"Finished processing {root_counter} ROOT files.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process ROOT files and save features/matrices to HDF5 format.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing ROOT files.")
    parser.add_argument("--max_events", type=int, default=None, help="Maximum number of events to process.")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="Directory to save HDF5 files.")
    parser.add_argument("--max_root_files", type=int, default=None, help="Maximum number of ROOT files to process.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Call the main processing function with parsed arguments
    process_root_files_to_parquet(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        max_root_files=args.max_root_files,
    )

    # load one example file and print contents
    example_file = next(output_dir.glob("*.parquet"))
    example_data = ak.from_parquet(example_file)

    logging.info(f"Contents of example file {example_file}:")
    for key in example_data.fields:
        logging.info(f"{key}: {example_data[key]}")
