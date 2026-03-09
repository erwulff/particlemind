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

        self.start_idx = start_idx
        self.stop_idx = stop_idx

        self.vit_kwargs = vit_kwargs

        self.BINS_X = np.linspace(
            -vit_kwargs["X_MAX"],
            vit_kwargs["X_MAX"],
            vit_kwargs["NUM_X_PATCHES"] * vit_kwargs["NUM_BINS_X_PATCH"] + 1,
        )
        self.BINS_Y = np.linspace(
            -vit_kwargs["Y_MAX"],
            vit_kwargs["Y_MAX"],
            vit_kwargs["NUM_Y_PATCHES"] * vit_kwargs["NUM_BINS_Y_PATCH"] + 1,
        )
        self.BINS_Z = np.linspace(
            -vit_kwargs["Z_MAX"],
            vit_kwargs["Z_MAX"],
            vit_kwargs["NUM_Z_PATCHES"] * vit_kwargs["NUM_BINS_Z_PATCH"] + 1,
        )

        self.NUM_TOTAL_PATCHES = (
            vit_kwargs["NUM_X_PATCHES"]
            * vit_kwargs["NUM_Y_PATCHES"]
            * vit_kwargs["NUM_Z_PATCHES"]
        )

        self.NUM_BINS_XYZ_PATCH = (
            vit_kwargs["NUM_BINS_X_PATCH"]
            * vit_kwargs["NUM_BINS_Y_PATCH"]
            * vit_kwargs["NUM_BINS_Z_PATCH"]
        )

        # --- Precompute for fast histogramming ---
        self.nx = len(self.BINS_X) - 1
        self.ny = len(self.BINS_Y) - 1
        self.nz = len(self.BINS_Z) - 1

        self.dx = self.BINS_X[1] - self.BINS_X[0]
        self.dy = self.BINS_Y[1] - self.BINS_Y[0]
        self.dz = self.BINS_Z[1] - self.BINS_Z[0]

        self.xmin = self.BINS_X[0]
        self.ymin = self.BINS_Y[0]
        self.zmin = self.BINS_Z[0]

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

            # -------------------------------------------------
            # FAST 3D histogram (replaces np.histogramdd)
            # -------------------------------------------------

            ix = ((x - self.xmin) / self.dx).astype(np.int32)
            iy = ((y - self.ymin) / self.dy).astype(np.int32)
            iz = ((z - self.zmin) / self.dz).astype(np.int32)

            mask = (
                (ix >= 0) & (ix < self.nx)
                & (iy >= 0) & (iy < self.ny)
                & (iz >= 0) & (iz < self.nz)
            )

            ix = ix[mask]
            iy = iy[mask]
            iz = iz[mask]
            w = energy[mask]

            flat_idx = ix * (self.ny * self.nz) + iy * self.nz + iz

            hist_flat = np.bincount(
                flat_idx,
                weights=w,
                minlength=self.nx * self.ny * self.nz,
            ).astype(np.float32)

            hist = hist_flat.reshape(self.nx, self.ny, self.nz)

            # Match previous density=True behavior (approximate)
            total = hist.sum()
            if total > 0:
                hist /= total

            # -------------------------------------------------

            hist_inputs = (
                hist.reshape(
                    self.vit_kwargs["NUM_X_PATCHES"],
                    self.vit_kwargs["NUM_BINS_X_PATCH"],
                    self.vit_kwargs["NUM_Y_PATCHES"],
                    self.vit_kwargs["NUM_BINS_Y_PATCH"],
                    self.vit_kwargs["NUM_Z_PATCHES"],
                    self.vit_kwargs["NUM_BINS_Z_PATCH"],
                )
                .transpose(0, 2, 4, 1, 3, 5)
                .reshape(
                    self.NUM_TOTAL_PATCHES,
                    self.NUM_BINS_XYZ_PATCH,
                )
            )

            hit_labels = np.array(event["detector"])

            yield {
                "hit_labels": hit_labels,
                "calo_hit_features": hist_inputs,
            }