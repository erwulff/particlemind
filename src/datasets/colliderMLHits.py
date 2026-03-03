import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset
import itertools
import logging



class colliderMLHits(IterableDataset):
    def __init__(
        self,
        subset,
        split,
        vit_kwargs,
        nsamples=None,
        train_fraction=0.8,
        start_idx=None,
        stop_idx=None,
    ):
        self.subset = subset
        self.split = split
        self.nsamples = nsamples
        self.train_fraction = train_fraction

        # NEW
        self.start_idx = start_idx
        self.stop_idx = stop_idx

        self.vit_kwargs = vit_kwargs

        self.BINS_X = np.linspace(-vit_kwargs["X_MAX"], vit_kwargs["X_MAX"], vit_kwargs["NUM_X_PATCHES"]*vit_kwargs["NUM_BINS_X_PATCH"]+1) 
        self.BINS_Y = np.linspace(-vit_kwargs["Y_MAX"], vit_kwargs["Y_MAX"], vit_kwargs["NUM_Y_PATCHES"]*vit_kwargs["NUM_BINS_Y_PATCH"]+1)
        self.BINS_Z = np.linspace(-vit_kwargs["Z_MAX"], vit_kwargs["Z_MAX"], vit_kwargs["NUM_Z_PATCHES"]*vit_kwargs["NUM_BINS_Z_PATCH"]+1)
        self.NUM_TOTAL_PATCHES = vit_kwargs["NUM_X_PATCHES"]*vit_kwargs["NUM_Y_PATCHES"]*vit_kwargs["NUM_Z_PATCHES"]
        self.NUM_BINS_XYZ_PATCH = vit_kwargs["NUM_BINS_X_PATCH"]*vit_kwargs["NUM_BINS_Y_PATCH"]*vit_kwargs["NUM_BINS_Z_PATCH"]

    def _get_stream(self):
        return load_dataset(
            "OpenDataDetector/ColliderML-Release-1",
            self.subset,
            split="train",
            streaming=True,
            columns=["event_id", "detector", "total_energy", "x", "y", "z"],
        )

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
            idx_in_split = i % 100
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

        
        
            # make a 3D histogram
            hist, edges = np.histogramdd(
                np.column_stack((x, y, z)),
                bins=(self.BINS_X, self.BINS_Y, self.BINS_Z),
                weights=energy,        
                density=True          
            )
        
            # reshape to (NUM_TOTAL_PATCHES, NUM_BINS_XYZ_PATCH)
            hist_inputs = (
                hist.reshape(
                    self.vit_kwargs["NUM_X_PATCHES"], self.vit_kwargs["NUM_BINS_X_PATCH"],
                    self.vit_kwargs["NUM_Y_PATCHES"], self.vit_kwargs["NUM_BINS_Y_PATCH"],
                    self.vit_kwargs["NUM_Z_PATCHES"], self.vit_kwargs["NUM_BINS_Z_PATCH"]
                )
                .transpose(0, 2, 4, 1, 3, 5)   # group patch indices first
                .reshape(
                    self.NUM_TOTAL_PATCHES,
                    self.NUM_BINS_XYZ_PATCH
                )
            )

            hit_labels = np.array(event["detector"])

            yield {
                "hit_labels": hit_labels,
                "calo_hit_features": hist_inputs,
            }