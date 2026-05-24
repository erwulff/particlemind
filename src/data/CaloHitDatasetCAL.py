import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset
import itertools
import logging


from src.data.augmentations import standardize_calo_hit_features_xyz, augment_data


class CaloHitDataset(IterableDataset):
    def __init__(
        self,
        subsets,
        split,
        nsamples=None,
        train_fraction=0.8,
        E_min=0.00075,
        start_idx=None,
        stop_idx=None,
        augment_dataset=False
    ):
        self.subsets = subsets
        self.split = split
        self.nsamples = nsamples
        self.train_fraction = train_fraction
        self.E_min = E_min
        self.augment_dataset = augment_dataset

        # NEW
        self.start_idx = start_idx
        self.stop_idx = stop_idx

    def _get_stream(self):
        def stream_with_subset(subset_name, subset_id):
            dataset = load_dataset(
                "CERN/ColliderML-Release-1",
                subset_name,
                split="train",
                streaming=True,
                columns=["event_id", "detector", "total_energy", "x", "y", "z"],
            )
            for event in dataset:
                event["subset"] = subset_name      # human-readable
                #event["subset_id"] = subset_id     # numeric (better for models)
                yield event

        streams = [
            stream_with_subset(subset, i)
            for i, subset in enumerate(self.subsets)
        ]

        return itertools.chain.from_iterable(zip(*streams))

    def __len__(self):
        # nsamples overrides everything
        if self.nsamples is not None:
            return self.nsamples
    
        # if slicing is defined
        if self.start_idx is not None and self.stop_idx is not None:
            return max(0, self.stop_idx - self.start_idx)
    
        # otherwise unknown
        raise TypeError("Length unknown for streaming dataset without nsamples or stop_idx")

    def __iter__(self):
        logger = logging.getLogger(__name__)

        dataset = self._get_stream()

        # ----------------------------
        # Apply global start/stop slice
        # ----------------------------
        #start_i = self.start_idx or 0

        if self.start_idx is not None:
            dataset = itertools.islice(dataset, self.start_idx, self.stop_idx)
            start_i = self.start_idx
        else:
            start_i = 0
       

        # ----------------------------
        # DDP rank/world
        # ----------------------------
        rank = 0
        world_size = 1
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        # ----------------------------
        # DataLoader workers
        # ----------------------------
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        sample_counter = 0

        # IMPORTANT:
        # enumerate starting from GLOBAL index
        for i, event in enumerate(dataset, start=start_i):

            # ----------------------------
            # deterministic train/val split
            # ----------------------------
            idx_in_split = hash((event["event_id"], tuple(self.subsets))) % 100
            is_train_event = idx_in_split < int(100 * self.train_fraction)

            if self.split == "train" and not is_train_event:
                continue
            if self.split == "val" and is_train_event:
                continue

            # ----------------------------
            # DDP + worker sharding
            # ----------------------------
            if (i % world_size) != rank:
                continue
            if ((i // world_size) % num_workers) != worker_id:
                continue

            # ----------------------------
            # nsamples limit
            # ----------------------------
            if self.nsamples is not None and sample_counter >= self.nsamples:
                return
            sample_counter += 1


            # ----------------------------
            # build features
            # ----------------------------
            x = np.array(event["x"], dtype=np.float32)
            y = np.array(event["y"], dtype=np.float32)
            z = np.array(event["z"], dtype=np.float32)

         
            energy = np.array(event["total_energy"], dtype=np.float32)



            mask = energy >= self.E_min
            if mask.sum() == 0:
                continue

            calo_hit_features = np.column_stack(
                (x[mask], y[mask], z[mask], energy[mask])
            )

            calo_hit_labels =  np.array(event["detector"])[mask]

            is_ECAL = np.isin(calo_hit_labels, [9, 10, 11])
            is_HCAL = np.isin(calo_hit_labels, [12, 13, 14])

            calo_hit_features_std = standardize_calo_hit_features_xyz(calo_hit_features)


            to_yield = {
                "labels_ECAL":calo_hit_labels[is_ECAL],
                "labels_HCAL":calo_hit_labels[is_HCAL],
                "calo_hit_features_HCAL": calo_hit_features_std[is_HCAL],
                "calo_hit_features_ECAL": calo_hit_features_std[is_ECAL],
                "subset": event["subset"]
            }


            if self.augment_dataset: 
                augmented_data = standardize_calo_hit_features_xyz(augment_data(calo_hit_features))
                to_yield["calo_hit_features_augmented"] = augmented_data

  

            yield to_yield
