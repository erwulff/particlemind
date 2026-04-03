import numpy as np
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
            "patch_positions": [],
        })

        # ------------------------------------------------------------
        # 1. group + convert once
        # ------------------------------------------------------------
        for inp in inputs:
            for key, obj in inp.items():

                #tmp =  np.log( * 1e2) / 10
                

                grouped[key]["flat_tensor"].append(
                    torch.as_tensor(obj["flat_tensor"], dtype=torch.float32)
                )
                grouped[key]["global_patch_ids"].append(
                    torch.as_tensor(obj["global_patch_ids"], dtype=torch.long)
                )
                grouped[key]["local_patch_ids"].append(
                    torch.as_tensor(obj["local_patch_ids"], dtype=torch.long)
                )
                grouped[key]["patch_positions"].append(
                    torch.as_tensor(obj["patch_positions"], dtype=torch.float32)
                )
        
        # ------------------------------------------------------------
        # 2. concat within each key (ragged batch flattening)
        # ------------------------------------------------------------
        out = {}

        for key, obj in grouped.items():
           
            out[key] = {
                "flat_tensor": torch.stack(obj["flat_tensor"], dim=0).to(torch.float32),  # (B, P, C)
                "global_patch_ids": torch.stack(obj["global_patch_ids"], dim=0).to(torch.long),      # (B, P)
                "local_patch_ids": torch.stack(obj["local_patch_ids"], dim=0).to(torch.long),      # (B, P, 3)
                "patch_positions": torch.stack(obj["patch_positions"], dim=0).to(torch.float32),      # (B, P, 3)
            }
            
            axis_sum = torch.sum(torch.abs(out[key]["flat_tensor"]), dim=2)
            out[key]["mask"] = torch.where(axis_sum > 0, 1.0, 0.0)

    


        return out