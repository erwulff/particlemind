import torch
from collections import defaultdict


class Collater:
    """
    Concatenate everything per key across batch.
    Fully ragged, no padding, no stacking.
    """

    def __call__(self, inputs):

        grouped = defaultdict(lambda: {
            "flat_tensor": [],
            "global_patch_ids": [],
            "local_patch_ids": [],
        })

        # ------------------------------------------------------------
        # 1. group + convert once
        # ------------------------------------------------------------
        for inp in inputs:
            for key, obj in inp.items():

                grouped[key]["flat_tensor"].append(
                    torch.as_tensor(obj["flat_tensor"], dtype=torch.float32)
                )
                grouped[key]["global_patch_ids"].append(
                    torch.as_tensor(obj["global_patch_ids"], dtype=torch.long)
                )
                grouped[key]["local_patch_ids"].append(
                    torch.as_tensor(obj["local_patch_ids"], dtype=torch.long)
                )
        
        # ------------------------------------------------------------
        # 2. concat within each key (ragged batch flattening)
        # ------------------------------------------------------------
        out = {}

        for key, obj in grouped.items():
           
            out[key] = {
                "flat_tensor": torch.stack(obj["flat_tensor"], dim=0),  # (B, P, C)
                "global_patch_ids": torch.stack(obj["global_patch_ids"], dim=0),      # (B, P)
                "local_patch_ids": torch.stack(obj["local_patch_ids"], dim=0),      # (B, P, 3)
            }

        return out