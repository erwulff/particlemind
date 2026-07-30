import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset
import itertools
import logging
from src.data.patching import assign_hits_to_patches, assign_hits_to_patches



class CaloPatchDataset(IterableDataset):
    def __init__(
        self,
        subsets,
        split,
        detector_type,
        patch_registry_barrel,
        patch_registry_endcap_pos,
        patch_registry_endcap_neg,
        detector_patching_params,
        nsamples=None,
        train_fraction=0.8,
        start_idx=None,
        stop_idx=None,
    ):
        self.subsets = subsets
        self.split = split
        self.nsamples = nsamples
        self.train_fraction = train_fraction

        self.start_idx = start_idx
        self.stop_idx = stop_idx

        assert detector_type in ["ECAL", "HCAL"], "detector_type must be either 'ECAL' or 'HCAL'"
        self.detector_type = detector_type
        self.patch_registry_barrel = patch_registry_barrel
        self.patch_registry_endcap_pos = patch_registry_endcap_pos
        self.patch_registry_endcap_neg = patch_registry_endcap_neg
        self.detector_patching_params = detector_patching_params

        # Per-region offsets so global patch IDs and local (ring, z) indices are
        # non-overlapping across barrel + two endcaps (required for deterministic argsort
        # and non-aliased positional encoding embeddings).
        self._n_barrel      = len(patch_registry_barrel["patches"])
        self._n_endcap_pos  = len(patch_registry_endcap_pos["patches"])

        self._r_off_endcap_pos = patch_registry_barrel["n_rings"]
        self._r_off_endcap_neg = patch_registry_barrel["n_rings"] + patch_registry_endcap_pos["n_rings"]

        self._z_off_endcap_pos = patch_registry_barrel["n_z_patches"]
        self._z_off_endcap_neg = patch_registry_barrel["n_z_patches"] + patch_registry_endcap_pos["n_z_patches"]



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

            if self.detector_type == "ECAL":
                det_index_barrel = 10
                det_index_pos_endcap = 11
                det_index_neg_endcap = 9
            elif self.detector_type == "HCAL":
                det_index_barrel = 13
                det_index_pos_endcap = 14
                det_index_neg_endcap = 12




            mask_barrel = hit_labels==det_index_barrel
            output_barrel = assign_hits_to_patches(
                x[mask_barrel],
                y[mask_barrel],
                z[mask_barrel],
                energy[mask_barrel],
                self.patch_registry_barrel,
                self.detector_patching_params
            ) 

            mask_endcap_pos = hit_labels==det_index_pos_endcap
            output_endcap_pos = assign_hits_to_patches(
                x[mask_endcap_pos],
                y[mask_endcap_pos],
                z[mask_endcap_pos],
                energy[mask_endcap_pos],
                self.patch_registry_endcap_pos,
                self.detector_patching_params
            )   

            mask_endcap_neg = hit_labels==det_index_neg_endcap
            output_endcap_neg = assign_hits_to_patches(
                x[mask_endcap_neg],
                y[mask_endcap_neg],
                z[mask_endcap_neg],
                energy[mask_endcap_neg],
                self.patch_registry_endcap_neg,
                self.detector_patching_params
            )

            # Combine barrel + endcaps into one dict.
            # Keys are prefixed ("barrel_", "endcap_pos_", "endcap_neg_") to avoid
            # collisions.  Ring and z local_patch_ids are shifted so the positional
            # encoding embedding tables see globally-unique indices across all three
            # regions.  global_patch_ids are shifted for the same reason (argsort).
            output = {}
            for region_tag, region_out, gid_off, r_off, z_off in [
                ("barrel",     output_barrel,     0,                                         0,                   0),
                # ("endcap_pos", output_endcap_pos, self._n_barrel,                            self._r_off_endcap_pos, self._z_off_endcap_pos),
                # ("endcap_neg", output_endcap_neg, self._n_barrel + self._n_endcap_pos,       self._r_off_endcap_neg, self._z_off_endcap_neg),
            ]:
                for k, v in region_out.items():
                    local_ids = v["local_patch_ids"].copy()
                    local_ids[:, 0] += r_off  # ring index
                    local_ids[:, 2] += z_off  # z index
                    output[f"{region_tag}_{k}"] = {
                        "flat_tensor":      v["flat_tensor"],
                        "global_patch_ids": v["global_patch_ids"] + gid_off,
                        "local_patch_ids":  local_ids,
                        "patch_positions":  v["patch_positions"],
                    }

            yield output