from pathlib import Path
import numpy as np
import awkward as ak

import torch
from torch.utils.data import IterableDataset
import random

import logging


def get_hit_labels(hit_idx, gen_idx, weights):
    """
    Assign labels to hits based on the genparticle index with the highest weight.

    Parameters:
        hit_idx (np.ndarray): Array of hit indices.
        gen_idx (np.ndarray): Array of genparticle indices corresponding to each hit.
        weights (np.ndarray): Array of weights corresponding to each hit.

    Returns:
        np.ndarray: Array of labels for each hit, where each label corresponds to the genparticle index.
    """
    # Initialize an array to store labels for each hit
    hit_labels = np.full(np.max(hit_idx) + 1, -1, dtype=int)  # Default label is -1 (unclassified)
    hit_label_weights = dict()  # To keep track of the highest weight for each hit

    # Iterate through the sparse COO matrix data
    for h_idx, g_idx, weight in zip(hit_idx, gen_idx, weights):
        if hit_labels[h_idx] == -1 or weight > hit_label_weights[h_idx]:
            hit_labels[h_idx] = g_idx
            hit_label_weights[h_idx] = weight

    # hit_labels now contains the genparticle index for each hit

    return hit_labels


class CLDHits(IterableDataset):
    def __init__(self, folder_path, split, nsamples=None, shuffle_files=False, train_fraction=0.8):
        """
        Initialize the dataset by storing the paths to all parquet files in the specified folder.

        Args:
            folder_path (str or Path): Path to the folder containing parquet files.
            shuffle_files (bool): Whether to shuffle the order of parquet files.
        """
        self.folder_path = Path(folder_path)
        self.parquet_files = list(self.folder_path.glob("*.parquet"))
        self.shuffle_files = shuffle_files
        self.nsamples = nsamples
        if self.nsamples is not None:
            self.sample_counter = 0

        self.split = split
        if self.split is not None:
            split_index = int(len(self.parquet_files) * train_fraction)
            if self.split == "train":
                self.parquet_files = self.parquet_files[:split_index]
            elif self.split == "val":
                self.parquet_files = self.parquet_files[split_index:]

        if self.shuffle_files:
            self.shuffle_shards()

    def __len__(self):
        """
        Return the number of events in the dataset.
        """
        data = ak.from_parquet(self.parquet_files[0])
        events_per_file = len(data[data.fields[0]])
        return len(self.parquet_files) * events_per_file if self.nsamples is None else self.nsamples

    def shuffle_shards(self):
        """
        Shuffle the parquet files. This can be called at the start of every epoch.
        """
        random.shuffle(self.parquet_files)

    def __iter__(self):
        logger = logging.getLogger(__name__)
        self.sample_counter = 0  # Reset sample counter for each iteration or each epoch
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            files_to_process = self.parquet_files
            logger.info(f"Processing {len(files_to_process)} files in single-process mode.")

        else:
            # Multi-process data loading, split the files among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files_to_process = self.parquet_files[worker_id::num_workers]
            logger.info(f"Processing {len(files_to_process)} files out of {len(self.parquet_files)} total files.")

        for file in files_to_process:
            data = ak.from_parquet(file)
            for event_i in range(len(data["genparticle_to_calo_hit_matrix"])):
                if self.nsamples is not None:
                    if self.sample_counter >= self.nsamples:
                        return
                    self.sample_counter += 1

                genparticle_to_calo_hit_matrix = data["genparticle_to_calo_hit_matrix"][event_i]
                calo_hit_features = data["calo_hit_features"][event_i]

                gen_idx = genparticle_to_calo_hit_matrix["gen_idx"].to_numpy()
                hit_idx = genparticle_to_calo_hit_matrix["hit_idx"].to_numpy()
                weights = genparticle_to_calo_hit_matrix["weight"].to_numpy()

                calo_hit_features = np.column_stack(
                    (
                        calo_hit_features["position.x"].to_numpy() / 1000,
                        calo_hit_features["position.y"].to_numpy() / 1000,
                        calo_hit_features["position.z"].to_numpy() / 1000,
                        calo_hit_features["energy"].to_numpy() * 100,
                    )
                )

                hit_labels = get_hit_labels(
                    hit_idx, gen_idx, weights
                )  # This could be moved to the pre-processing step if needed

                yield {
                    # "gen_idx": gen_idx,
                    # "hit_idx": hit_idx,
                    # "weights": weights,
                    "hit_labels": hit_labels,
                    "calo_hit_features": calo_hit_features,
                }


# adapted from from: https://github.com/jpata/particleflow/blob/a3a08fe1e687987c661faad00fd5526e733be014/mlpf/model/PFDataset.py#L163
class Collater:
    """
    Custom collator for DataLoader to handle variable-sized inputs.
    This collator pads variable-sized inputs and stacks fixed-size inputs.
    It is designed to work with datasets where some features (like particle hits) can vary in size,
    while others (like event-level features) are fixed-size.
    Args:
        variable_size_keys (list): List of keys for variable-sized inputs that need padding.
        fixed_size_keys (list): List of keys for fixed-sized inputs that can be stacked.
    Returns:
        dict: A dictionary containing padded and stacked inputs.
    """

    def __init__(self, variable_size_keys="all", fixed_size_keys=None, **kwargs):
        super(Collater, self).__init__(**kwargs)
        self.variable_size_keys = variable_size_keys
        self.fixed_size_keys = fixed_size_keys

    def __call__(self, inputs):
        ret = {}

        if self.variable_size_keys == "all":
            for key in inputs[0].keys():
                ret[key] = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(inp[key]).to(torch.float32) for inp in inputs], batch_first=True
                )
            return ret

        # per-particle quantities need to be padded across events of different size
        for key_to_get in self.variable_size_keys:
            ret[key_to_get] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(inp[key_to_get]).to(torch.float32) for inp in inputs], batch_first=True
            )

        # per-event quantities can be stacked across events
        if self.fixed_size_keys:
            for key_to_get in self.fixed_size_keys:
                ret[key_to_get] = torch.stack([torch.tensor(inp[key_to_get]) for inp in inputs])
        return ret


def collate_fn(batch):
    """
    Expects downstream code to handle variable sizes.

    Args:
        batch (list): A list of dictionaries, where each dictionary represents an event.

    Returns:
        list: A list of events (unchanged) for variable-sized data.
    """
    return batch
