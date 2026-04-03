
import numpy as np
import torch
from collections import defaultdict


class CollaterPatch:
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


# adapted from from: https://github.com/jpata/particleflow/blob/a3a08fe1e687987c661faad00fd5526e733be014/mlpf/model/PFDataset.py#L163
class CollaterHits:
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

    def __init__(self, empty_key, variable_size_keys="all", fixed_size_keys=None, pad=-1, **kwargs):
        super(Collater, self).__init__(**kwargs)
        self.variable_size_keys = variable_size_keys
        self.fixed_size_keys = fixed_size_keys
        self.empty_key = empty_key
        self.pad = pad

    def __call__(self, inputs):
        ret = {}

        if self.variable_size_keys == "all":
            for key in inputs[0].keys():

                if self.pad > 0:
                    ret[key] = torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(inp[key][:self.pad]).to(torch.float32) for inp in inputs], batch_first=True
                    )
                else: 
                    ret[key] = torch.nn.utils.rnn.pad_sequence(
                        [torch.tensor(inp[key]).to(torch.float32) for inp in inputs], batch_first=True
                    )

            # get mask
            axis_sum = torch.sum(torch.abs(ret[self.empty_key]), dim=2)
            ret["mask"] = torch.where(axis_sum > 0, 1.0, 0.0)
    

            return ret

        """
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

        """
