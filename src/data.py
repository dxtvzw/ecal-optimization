from typing import Any
from torch.utils.data import DataLoader, Dataset
import uproot
import numpy as np
from pathlib import Path
import re

import random

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

from utils import Timer


def extract_dimensions(filename):
    match = re.search(r'\d+x\d+', filename)
    if match:
        dimensions = match.group(0)
        x, y = dimensions.split('x')
        return int(x), int(y)
    else:
        return None


def get_data(cfg, logger):
    timer = Timer()
    logger.info("Loading the data")

    height, width = extract_dimensions(cfg.paths.data_file)

    cfg.data.height = height
    cfg.data.width = width

    root_file = uproot.open(Path(cfg.paths.data_dir) / cfg.paths.data_file)
    nevents=None

    RawE = root_file['ecalNT']['RawEnergyDeposit'].arrays(library='np')['RawEnergyDeposit'][:nevents]
    x = root_file['ecalNT']['RawX'].arrays(library='np')['RawX'][:nevents]
    y = root_file['ecalNT']['RawY'].arrays(library='np')['RawY'][:nevents]
    z = root_file['ecalNT']['RawZ'].arrays(library='np')['RawZ'][:nevents]

    EnergyDeposit = np.array(root_file['ecalNT']['EnergyDeposit'].array()[:nevents])
    EnergyDeposit = EnergyDeposit.reshape(-1, height, width)[:, None, :, :]

    ParticlePDG = np.array(root_file['ecalNT']['ParticlePDG'].array())[:nevents]
    ParticleMomentum_v = np.array(root_file['ecalNT']['ParticleMomentum'].array())[:nevents]
    ParticleMomentum = np.sum(ParticleMomentum_v * ParticleMomentum_v, axis=1) ** 0.5

    X = np.array(EnergyDeposit)
    y = ParticleMomentum

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.data.test_size, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    train_generator = torch.Generator()
    train_generator.manual_seed(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.data.num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.data.num_workers,
    )

    logger.info(f"Successfully loaded the data || {timer()}")

    return train_loader, val_loader
