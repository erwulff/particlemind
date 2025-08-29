# particlemind
Self-supervised learning on HEP events.

## Datasets

A small testing dataset (about 20GB) can be downloaded from zenodo:
```
./scripts/download_data.sh
```

### Approximate repo structure
```
├── README.md
├── configs              # Config files (models and training hyperparameters)
│   └── model1.yaml
├── models               # Trained and exported models.
│
├── notebooks            # Jupyter notebooks.
│
├── requirements.txt     # The requirements file for reproducing the environment.
└── src                  # Source code for use in this project.
    ├── __init__.py      # Makes src a Python module.
    │
    ├── datasets         # Data engineering scripts.
    │   ├── dataset1.py
    │   ├── dataset2.py
    │   └── utils.py.py
    │
    ├── models           # ML model engineering (a folder/file for each model).
    │   ├── model1.py
    │   ├── model2.py
    │   └── losses.py
    │
    └── scripts          # Various scripts for e.g., GPU-cluster training etc.
        ├── HPC1
        │   ├── train_4GPUs.sh
        │   └── train_8GPUs.sh
        └── HPC2
            ├── train_4GPUs.sh
            └── train_8GPUs.sh

# Data is assumed to be located in directories outside this repo
├── data                 # Data (not in git)
    ├── dataset1
    │   ├── raw
    │   └── processed
    └── dataset2
        ├── raw
        └── processed
```
