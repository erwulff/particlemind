from pathlib import Path
import numpy as np

from datasets import load_dataset

import torch
from torch.utils.data import IterableDataset
import random

import logging




def standardize_calo_hit_features(calo_hit_features):
    calo_hit_features[..., 0] = calo_hit_features[..., 0] / 1e4  # position x
    calo_hit_features[..., 1] = calo_hit_features[..., 1] / 1e4  # position y
    calo_hit_features[..., 2] = calo_hit_features[..., 2] / 1e4  # position z
    calo_hit_features[..., 3] = np.log(calo_hit_features[..., 3] * 1e2) / 10  # energy
    return calo_hit_features


def inverse_standardize_calo_hit_features(calo_hit_features):
    calo_hit_features[..., 0] = calo_hit_features[..., 0] * 1e4  # position x
    calo_hit_features[..., 1] = calo_hit_features[..., 1] * 1e4  # position y
    calo_hit_features[..., 2] = calo_hit_features[..., 2] * 1e4  # position z
    calo_hit_features[..., 3] = np.exp(calo_hit_features[..., 3] * 10) / 1e2  # energy
    return calo_hit_features


class colliderMLHits(IterableDataset):
    def __init__(
        self, 
        subset,
        split, 
        nsamples=None, 
        train_fraction=0.8, 
        E_min=0,
    ):
        """
        Initialize the dataset by storing the paths to all parquet files in the specified folder.

        Args:
            folder_path (str or Path): Path to the folder containing parquet files.
            shuffle_files (bool): Whether to shuffle the order of parquet files.
        """
        self.subset = subset
        self.split = split
        self.nsamples = nsamples
        self.train_fraction = train_fraction
        self.E_min = E_min
        


    def _get_stream(self):
        # streaming dataset (no local download)
        return load_dataset(
            "OpenDataDetector/ColliderML-Release-1",
            self.subset,
            split="train",  # only split available
            streaming=True,
            columns=["event_id", "detector", "total_energy", "x", "y", "z"],
        )


    def __len__(self):
       # TODO
       pass


    def __iter__(self):
        logger = logging.getLogger(__name__)
    
        dataset = self._get_stream()

        # --- CHANGE 1: get DDP rank & world_size --- #
        rank = 0
        world_size = 1
        if torch.distributed.is_initialized():  # check if DDP is initialized
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()


         # --- CHANGE 2: get DataLoader worker info --- #
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        sample_counter = 0

        for i, event in enumerate(dataset):
            # deterministic manual train/val split
            idx_in_split = i % 100  # simple modulo-based splitting
            is_train_event = idx_in_split < int(100 * self.train_fraction)
            if self.split == "train" and not is_train_event:
                continue
            if self.split == "val" and is_train_event:
                continue

            # DDP + worker sharding
            if (i % world_size) != rank:
                continue
            if ((i // world_size) % num_workers) != worker_id:
                continue

            if self.nsamples is not None and sample_counter >= self.nsamples:
                return
            sample_counter += 1

            x = np.array(event["x"])
            y = np.array(event["y"])
            z = np.array(event["z"])
            energy = np.array(event["total_energy"])

            mask = energy >= self.E_min
            calo_hit_features = np.column_stack((x[mask], y[mask], z[mask], energy[mask]))

            hit_labels = np.array(event["detector"])[mask]

            yield {
                # "gen_idx": gen_idx,
                # "hit_idx": hit_idx,
                # "weights": weights,
                "hit_labels": hit_labels,
                "calo_hit_features": standardize_calo_hit_features(calo_hit_features),
            }

            
