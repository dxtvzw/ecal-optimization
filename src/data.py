from typing import Any
from torch.utils.data import DataLoader, Dataset
import uproot
import numpy as np
from pathlib import Path
import re

import random

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import v2
import torch.nn as nn

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


def read_file(cfg):
    if cfg.paths.data_file.endswith("root"):
        root_file = uproot.open(Path(cfg.paths.data_dir) / cfg.paths.data_file)
        nevents=None

        RawE = root_file['ecalNT']['RawEnergyDeposit'].arrays(library='np')['RawEnergyDeposit'][:nevents]
        x = root_file['ecalNT']['RawX'].arrays(library='np')['RawX'][:nevents]
        y = root_file['ecalNT']['RawY'].arrays(library='np')['RawY'][:nevents]
        z = root_file['ecalNT']['RawZ'].arrays(library='np')['RawZ'][:nevents]

        EnergyDeposit = np.array(root_file['ecalNT']['EnergyDeposit'].array()[:nevents])
        EnergyDeposit = EnergyDeposit.reshape(-1, cfg.data.height, cfg.data.width)[:, None, :, :]

        ParticlePDG = np.array(root_file['ecalNT']['ParticlePDG'].array())[:nevents]
        ParticleMomentum_v = np.array(root_file['ecalNT']['ParticleMomentum'].array())[:nevents]
        ParticleMomentum = np.sum(ParticleMomentum_v * ParticleMomentum_v, axis=1) ** 0.5

        ParticlePoint = np.array(root_file['ecalNT']['ParticlePoint'].array())[:nevents]

        X = np.array(EnergyDeposit)
        y = np.concatenate((ParticleMomentum[:, None], ParticlePoint[:, :2]), axis=1)
    else:
        data = np.load(Path(cfg.paths.data_dir) / cfg.paths.data_file)
        X = data['X']
        y = data['y']

    return X, y


class MyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        data = self.X[index]
        trg = self.y[index]
        if self.transform is not None:
            data, trg = self.transform(data, trg)
        return data, trg


class Normalizer:
    def __init__(self):
        self.min_val = np.inf
        self.max_val = -np.inf
    
    def fit(self, arr: np.array):
        self.min_val = arr.min()
        self.max_val = arr.max()
    
    def transform(self, arr: np.array):
        res = (arr - self.min_val) / (self.max_val - self.min_val)
        return res

    def fit_transform(self, arr: np.array):
        self.fit(arr)
        return self.transform(arr)


class MyRandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x, y):
        assert x.shape[-1] % 2 == 1, "The shape of the image must be odd"

        if random.random() < self.p:
            x = x.flip(-1)
            y[2] = 1 - y[2]
        return x, y


class MyRandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x, y):
        assert x.shape[-1] % 2 == 1, "The shape of the image must be odd"

        if random.random() < self.p:
            x = x.flip(-2)
            y[1] = 1 - y[1]
        return x, y


class MyRandomRotation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        assert x.shape[-1] % 2 == 1, "The shape of the image must be odd"
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, (1, 2))
        if k == 0:
            new_y1, new_y2 = y[1], y[2]
        elif k == 1:
            new_y1, new_y2 = 1 - y[2], y[1]
        elif k == 2:
            new_y1, new_y2 = 1 - y[1], 1 - y[2]
        elif k == 3:
            new_y1, new_y2 = y[2], 1 - y[1]
        new_y1 = torch.clone(new_y1)
        new_y2 = torch.clone(new_y2)
        y[1] = new_y1
        y[2] = new_y2
        return x, y


def get_data(cfg, logger):
    timer = Timer()
    logger.info("Loading the data")

    height, width = extract_dimensions(cfg.paths.data_file)

    cfg.data.height = height
    cfg.data.width = width

    cfg.model.height = height
    cfg.model.width = width

    X, y = read_file(cfg)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.data.test_size, random_state=42)

    if cfg.data.normalize_position:
        x_normalizer = Normalizer()
        y_normalizer = Normalizer()

        x_normalizer.fit(y_train[:, 1])
        y_train[:, 1] = x_normalizer.transform(y_train[:, 1])
        y_val[:, 1] = x_normalizer.transform(y_val[:, 1])

        y_normalizer.fit(y_train[:, 2])
        y_train[:, 2] = y_normalizer.transform(y_train[:, 2])
        y_val[:, 2] = y_normalizer.transform(y_val[:, 2])

    if (cfg.data.transforms.use_flips or cfg.data.transforms.use_rotation) and height % 2 == 0:
        logger.info("The sensor shape is even, so the flips and rotations will not be applied")
        cfg.data.transforms.use_flips = False
        cfg.data.transforms.use_rotation = False

    train_transforms = []
    if cfg.data.transforms.use_flips:
        train_transforms.append(MyRandomHorizontalFlip(p=0.5))
        train_transforms.append(MyRandomVerticalFlip(p=0.5))
    if cfg.data.transforms.use_rotation:
        train_transforms.append(MyRandomRotation())

    train_transform = v2.Compose(train_transforms) if train_transforms else None

    train_dataset = MyDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        transform=train_transform,
    )
    val_dataset = MyDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

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
