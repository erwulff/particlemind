
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
import datasets
datasets.disable_caching()

import numpy as np
from tqdm import tqdm
import pickle

particles = load_dataset(
    "CERN/ColliderML-Release-1",
    "ttbar_pu0_calo_hits",
    split="train",
    streaming=True,
).select_columns(["x", "y", "z","total_energy", "detector"])

import gc
import os

NUM_EVENTS = 3000
CHUNK_SIZE = 500  # flush every this many events; tune down if kernel still OOMs

OUT_PATH = "/scratch/midway3/rmastand/particlemind/small_calo_hits.pkl"
CHUNK_DIR = "/scratch/midway3/rmastand/particlemind/chunks"
os.makedirs(CHUNK_DIR, exist_ok=True)

def empty_buf():
    return {"x": [], "y": [], "z": [], "r": [], "e": [], "detector": []}

def flush_chunk(buf, chunk_idx):
    path = os.path.join(CHUNK_DIR, f"chunk_{chunk_idx:04d}.pkl")
    arr = {k: np.array(v) for k, v in buf.items()}
    pickle.dump(arr, open(path, "wb"))
    return path

chunk_paths = []
chunk_idx = 0
buf = empty_buf()

for i, event in enumerate(tqdm(particles, total=NUM_EVENTS)):
    if i >= NUM_EVENTS:
        break

    buf["x"].extend(event["x"])
    buf["y"].extend(event["y"])
    buf["z"].extend(event["z"])
    buf["r"].extend(np.sqrt(np.array(event["x"])**2 + np.array(event["y"])**2))
    buf["e"].extend(event["total_energy"])
    buf["detector"].extend(event["detector"])

    if (i + 1) % CHUNK_SIZE == 0:
        chunk_paths.append(flush_chunk(buf, chunk_idx))
        chunk_idx += 1
        buf = empty_buf()
        gc.collect()

if any(buf[k] for k in buf):
    chunk_paths.append(flush_chunk(buf, chunk_idx))
    del buf
    gc.collect()

print(f"Wrote {len(chunk_paths)} chunk(s)")

# Merge chunks → numpy arrays in final output (supports boolean indexing downstream)
keys = ["x", "y", "z", "r", "e", "detector"]
data_dir = {k: np.concatenate([pickle.load(open(p, "rb"))[k] for p in chunk_paths]) for k in keys}

pickle.dump(data_dir, open(OUT_PATH, "wb"))

for path in chunk_paths:
    os.remove(path)

print(f"Saved {len(data_dir['x'])} hits → {OUT_PATH}")