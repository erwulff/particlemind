import uproot
import awkward as ak
import argparse
import logging
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

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


def process_single_root_file(root_file, output_dir):
    try:
        output_file = output_dir / f"{root_file.stem}.parquet"
        if output_file.exists():
            logging.info(f"Output file {output_file} already exists. Skipping processing.")
            return

        fi = uproot.open(root_file)
        collectionIDs = {
            k: v
            for k, v in zip(
                fi.get("podio_metadata").arrays("events___idTable/m_names")["events___idTable/m_names"][0],
                fi.get("podio_metadata").arrays("events___idTable/m_collectionIDs")["events___idTable/m_collectionIDs"][
                    0
                ],
            )
        }
        ev = fi["events"]
        event_data = get_event_data(ev)

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

        for iev in range(len(ev["MCParticles.momentum.x"].array())):
            gen_features = gen_to_features(event_data, iev)
            track_features = track_to_features(event_data, iev)
            cluster_features = cluster_to_features(
                event_data, iev, cluster_features=["position.x", "position.y", "position.z", "energy", "type"]
            )
            calo_hit_features, genparticle_to_calo_hit_matrix, _ = process_calo_hit_data(event_data, iev, collectionIDs)
            tracker_hit_features, genparticle_to_tracker_hit_matrix, _ = process_tracker_hit_data(
                event_data, iev, collectionIDs
            )
            track_to_tracker_hit_matrix, _ = create_track_to_hit_coo_matrix(event_data, iev, collectionIDs)
            cluster_to_cluster_hit_matrix, _ = create_cluster_to_hit_coo_matrix(event_data, iev, collectionIDs)
            gp_to_track_matrix = genparticle_track_adj(event_data, iev)
            gp_to_gp = create_genparticle_to_genparticle_coo_matrix(event_data, iev)

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

        for key in combined_data_dict.keys():
            combined_data_dict[key] = ak.Array(combined_data_dict[key])

        ak.to_parquet(combined_data_dict, output_file)

    except Exception as e:
        logging.error(f"Error processing {root_file}: {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process ROOT files and save features/matrices to parquet format.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing ROOT files.")
    parser.add_argument("--max_events", type=int, default=None, help="Maximum number of events to process.")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="Directory to save parquet files.")
    parser.add_argument("--max_root_files", type=int, default=None, help="Maximum number of ROOT files to process.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    max_root_files = args.max_root_files
    if not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory.")

    root_file_list = list(Path(input_dir).rglob("*.root"))
    total_files_to_process = max_root_files or len(root_file_list)

    with ProcessPoolExecutor() as executor:
        futures = []
        for root_file in sorted(root_file_list)[:total_files_to_process]:
            futures.append(executor.submit(process_single_root_file, root_file, output_dir))

        for future in tqdm(futures, desc="Processing ROOT files"):
            future.result()

    # load one example file and print contents
    example_file = next(output_dir.glob("*.parquet"))
    example_data = ak.from_parquet(example_file)

    logging.info(f"Contents of example file {example_file}:")
    for key in example_data.fields:
        logging.info(f"{key}: {example_data[key]}")
