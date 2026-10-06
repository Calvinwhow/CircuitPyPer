"""Composite atrophy-map construction."""

import pandas as pd
from typing import Tuple
from itertools import combinations
import numpy as np
import copy

def generate_tensor(dict):
    arrays = [np.asarray(dict[k]) for k in dict.keys()]
    voxels, patients = arrays[0].shape
    V = len(arrays)
    tensor = np.empty((voxels, patients, V), dtype=arrays[0].dtype)
    for i, arr in enumerate(arrays):
        tensor[:, :, i] = arr  # shape (voxels, patients, V)
    return tensor

def generate_norm(arr, atrophy_only=True):
    """Generates the L2 norm of a tensor"""
    if atrophy_only:
        arr = np.where(arr < 0, arr, 0)  # Only consider negative values for atrophy
    return np.sqrt(np.sum(arr**2, axis=2))

def prepocess_dict(d: dict) -> dict:
    d = copy.deepcopy(d)
    d.pop('white_matter', None)
    if 'cerebrospinal_fluid' in d:
        d['cerebrospinal_fluid'] = -d['cerebrospinal_fluid']
    return d

def generate_norm_map(pt_dict, ctrl_dict) -> pd.DataFrame:
    """Generates a composite atrophy map with L2 norm of Z scores"""
    pt_dict_processed = prepocess_dict(pt_dict)
    ctrl_dict_processed = prepocess_dict(ctrl_dict)
            
    ctrl_tensor = generate_tensor(ctrl_dict_processed)
    pt_tensor = generate_tensor(pt_dict_processed)
    ctrl_norm = generate_norm(ctrl_tensor)
    pt_norm = generate_norm(pt_tensor)
    
    mean = ctrl_norm.mean(axis=1)
    std  = ctrl_norm.std(axis=1)
    z = (pt_norm - mean[:, np.newaxis]) / std[:, np.newaxis]
    
    first_key = next(iter(pt_dict))
    z = pd.DataFrame(z, columns=pt_dict[first_key].columns, index=pt_dict[first_key].index)
    return z, mean, std

def generate_composite(dataframes: dict, csfgm_only: bool = True, sign_flip_csf: bool = False) -> dict:
    """
    Combine DataFrames to create composite maps.
    Optionally flip CSF or other segment signs.
    If csfgm_only is True, only combine CSF and GM; otherwise, combine all possible subsets.
    """
    dfs = dataframes.copy()
    if sign_flip_csf:
        print("Flipping CSF sign.")
        if 'cerebrospinal_fluid' in dfs:
            dfs['cerebrospinal_fluid'] = -dfs['cerebrospinal_fluid']
    else:
        print("Flipping GM and WM sign.")
        for k in dfs:
            if k != 'cerebrospinal_fluid':
                dfs[k] = -dfs[k]

    if csfgm_only:
        if 'cerebrospinal_fluid' in dfs and 'grey_matter' in dfs:
            dfs['csf+gm'] = dfs['cerebrospinal_fluid'] + dfs['grey_matter']
    else:
        keys = list(dfs.keys())
        for r in range(2, len(keys) + 1):
            for combo in combinations(keys, r):
                name = '+'.join(combo)
                dfs[name] = sum(dfs[k] for k in combo)
    return dfs
