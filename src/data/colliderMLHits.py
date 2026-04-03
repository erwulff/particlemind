import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset
import itertools
import logging
from src.data.patching import assign_hits_to_patches_barrel



class colliderMLHits(IterableDataset):
    def __init__(
        self,
        subset,
        split,
        patch_registry,
        detector_patching_params,
        nsamples=None,
        train_fraction=0.8,
        start_idx=None,
        stop_idx=None,
    ):
        self.subset = subset
        self.split = split
        self.nsamples = nsamples
        self.train_fraction = train_fraction

        self.start_idx = start_idx
        self.stop_idx = stop_idx

        self.patch_registry = patch_registry
        self.detector_patching_params = detector_patching_params



    def _get_stream(self):
        return load_dataset(
            "CERN/ColliderML-Release-1",
            self.subset,
            split="train",
            streaming=True,
            
            columns=["detector", "total_energy", "x", "y", "z"],
        )

    def __len__(self):
        if self.nsamples is not None:
            return self.nsamples

        if self.start_idx is not None and self.stop_idx is not None:
            return max(0, self.stop_idx - self.start_idx)

        raise TypeError(
            "Length unknown for streaming dataset without nsamples or stop_idx"
        )

    def __iter__(self):
        logger = logging.getLogger(__name__)

        dataset = self._get_stream()

        if self.start_idx is not None:
            dataset = itertools.islice(dataset, self.start_idx, self.stop_idx)
            start_i = self.start_idx
        else:
            start_i = 0

        rank = 0
        world_size = 1
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        sample_counter = 0

        for i, event in enumerate(dataset, start=start_i):
            
            if not all(k in event for k in ["x", "y", "z", "total_energy", "detector"]):
                continue

            idx_in_split = i % 100
            is_train_event = idx_in_split < int(100 * self.train_fraction)

            if self.split == "train" and not is_train_event:
                continue
            if self.split == "val" and is_train_event:
                continue

            if (i % world_size) != rank:
                continue
            if ((i // world_size) % num_workers) != worker_id:
                continue

            if self.nsamples is not None and sample_counter >= self.nsamples:
                return
            sample_counter += 1

            x = np.asarray(event["x"], dtype=np.float32)
            y = np.asarray(event["y"], dtype=np.float32)
            z = np.asarray(event["z"], dtype=np.float32)
            energy = np.asarray(event["total_energy"], dtype=np.float32)

        

            hit_labels = np.array(event["detector"])
            mask = hit_labels==13

            output = assign_hits_to_patches_barrel(
                x[mask],
                y[mask],
                z[mask],
                energy[mask],
                self.patch_registry,
                self.detector_patching_params
            ) 

        


            yield output