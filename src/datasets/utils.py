import numpy as np
import torch


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

    def __init__(self, variable_size_keys="all", fixed_size_keys=None, pad=-1, **kwargs):
        super(Collater, self).__init__(**kwargs)
        self.variable_size_keys = variable_size_keys
        self.fixed_size_keys = fixed_size_keys
        self.pad = pad

    def __call__(self, inputs):
        ret = {}

        if self.variable_size_keys == "all":
            for key in inputs[0].keys():
                ret[key] = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(inp[key]).to(torch.float32) for inp in inputs], batch_first=True
                )

            # get mask
            axis_sum = torch.sum(torch.abs(ret["calo_hit_features"]), dim=2)
            ret["calo_hit_mask"] = torch.where(axis_sum > 0, 1.0, 0.0)

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
