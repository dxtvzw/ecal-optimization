import numpy as np
import shutil, os
import uproot
from pathlib import Path
from tqdm.auto import tqdm
import re


def extract_dimensions(filename):
    match = re.search(r'\d+x\d+', filename)
    if match:
        dimensions = match.group(0)
        x, y = dimensions.split('x')
        return int(x), int(y)
    else:
        return None


if __name__ == '__main__':
    DATA_DIR = "data"
    NUMPY_DATA_DIR = "data_numpy"
    if not os.path.exists(NUMPY_DATA_DIR):
        os.makedirs(NUMPY_DATA_DIR)

    files = {
        "64k_real_15x15": "0001_64k_real_spectra_15x15_spot.root",
        "500k_real_15x15": "0001_500k_real_spectra_15x15_spot.root",
        "64k_wpc_10x10": "0001_64k_wpc_10x10_spot.root",
        "64k_wpc_10x10_v2": "0001_64k_wpc_10x10_spot_v2.root",
        "64k_wpc_15x15": "0001_64k_wpc_15x15_spot.root",
        "64k_wpc_20x20": "0001_64k_wpc_20x20_spot.root",
        "64k_wpc_25x25": "0001_64k_wpc_25x25_spot_rs42.root",
        "64k_wpc_30x30": "0001_64k_wpc_30x30_spot_rs42.root",
        "64k_wpc_40x40": "0001_64k_wpc_40x40_spot_rs42.root",
    }

    for key, value in tqdm(files.items()):
        assert extract_dimensions(key) == extract_dimensions(value)

        height, width = extract_dimensions(value)

        root_file = uproot.open(Path(DATA_DIR) / value)
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

        np.savez(Path(NUMPY_DATA_DIR) / key, X=X, y=y)
