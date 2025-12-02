from pathlib import Path
import numpy as np
import awkward as ak

import torch
from torch.utils.data import IterableDataset
import random

import logging


class Tokens(IterableDataset):
    def __init__(
        self, folder_path, split, nsamples=None, shuffle_files=False, train_fraction=0.8, nfiles=-1, by_event=True
    ):
        """
        Initialize the dataset by storing the paths to all parquet files in the specified folder.

        Args:
            folder_path (str or Path): Path to the folder containing parquet files.
            shuffle_files (bool): Whether to shuffle the order of parquet files.
        """
        self.folder_path = Path(folder_path)
        self.parquet_files = list(self.folder_path.glob("*.parquet"))
        self.shuffle_files = shuffle_files
        self.nsamples = nsamples
        if self.nsamples is not None:
            self.sample_counter = 0
        self.nfiles = nfiles
        self.by_event = by_event

        self.split = split
        if self.split is not None:
            split_index = int(len(self.parquet_files) * train_fraction)
            if self.split == "train":
                self.parquet_files = self.parquet_files[:split_index]
            elif self.split == "val":
                self.parquet_files = self.parquet_files[split_index:]

        if self.shuffle_files:
            self.shuffle_shards()


    """
    def __len__(self):
       
        #Return the number of events in the dataset.
        
        data = ak.from_parquet(self.parquet_files[0])
        events_per_file = len(data[data.fields[0]])
        return len(self.parquet_files) * events_per_file if self.nsamples is None else self.nsamples
    """

    def shuffle_shards(self):
        """
        Shuffle the parquet files. This can be called at the start of every epoch.
        """
        random.shuffle(self.parquet_files)

    def __iter__(self):
        logger = logging.getLogger(__name__)
        self.sample_counter = 0  # Reset sample counter for each iteration or each epoch
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process data loading
            files_to_process = self.parquet_files[: self.nfiles]
            logger.info(f"Processing {len(files_to_process)} files in single-process mode.")

        else:
            # Multi-process data loading, split the files among workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files_to_process = self.parquet_files[worker_id::num_workers]
            logger.info(f"Processing {len(files_to_process)} files out of {len(self.parquet_files)} total files.")


        for file in files_to_process:
            data = ak.from_parquet(file)
         
            # from backbone.model_step:
                # all token-ids up to the last one are the input, the ones from the second
                # to the (including) last one are the target
                # this model step uses the convention that the first particle feature
                # is the token, with the tokens up to the last one
                # the second particle feature is the target token (i.e. the next token)

            for event_i in range(len(data)):
                if self.nsamples is not None:
                    if self.sample_counter >= self.nsamples:
                        return
                    self.sample_counter += 1


                token_i = data[event_i]

                
                token_features = np.column_stack(
                    (
                        token_i[:-1].to_numpy(),
                        token_i[1:].to_numpy(),
                    )
                )


                if self.by_event:
                    yield {
                        "token_features": token_features,
                    }

                else:
                    # return one hit at a time instead of one event
                    for i in range(len(calo_hit_features)):
                        if self.nsamples is not None and self.sample_counter >= self.nsamples:
                            return
                        self.sample_counter += 1

                        yield {
                            "token_features": token_features[i : i + 1],  # Shape (1,) or (1, label_dim)
                        }

