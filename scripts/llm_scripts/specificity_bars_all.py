#!/usr/bin/env python3
"""
Symptom-specificity bars for every cluster / a priori regression under SCA_ROOT.

For every contrast map: score each patient as sum(patient Nifti_File_Path image x FWE map)
over the 2mm brain mask (maps on another grid are resampled to it), Spearman-correlate
the score with every symptom, and plot rhos grouped motor | cognitive | emotional in one
fixed symptom order. One folder of SVGs + a rho CSV + a PNG contact sheet per regression.

This is not a CLI. Change the values in the CONFIG section, then run.
Optional argv: substrings; only regressions whose path contains one of them are run.
"""
import os, sys, glob, re, json, io
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy.stats import spearmanr
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
for p in (str(CIRCUIT_PYPER_DIR.parent), str(CIRCUIT_PYPER_DIR)):
    if p not in sys.path: sys.path.insert(0, p)
from calvin_utils.neuroimaging_utils.ccm_utils.symptom_specificity import SpecificityAnalyzer

# =============================================================================
# CONFIG
# =============================================================================
SCA_ROOT = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/symptom_on_lhs"
SHEET = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/optimized_master_list_filtered_HigherIsWorse.csv"
BIDS_PREFIX = None           # (old, new) to rewrite image paths when the drive is mounted elsewhere
IMAGE_COLUMN = "Nifti_File_Path"
MASK = str(CIRCUIT_PYPER_DIR / "resources" / "MNI152_T1_2mm_brain_mask.nii")
LABELS_FROM = "vlsm_regressions_clusters/cluster_regression_input.csv"   # symptom -> domain
RUN_GLOB = "*/*/*-on-*/regression"          # a regression's output folder
RUN_KEYWORDS = ("cluster", "apriori")        # which trees count as "a cluster we made"
OUT_DIR = os.path.join(SCA_ROOT, "specificity_all")
DOMAIN_ORDER = ["motor", "cognitive", "emotional"]
# =============================================================================


def resample(img, ref):
    d = np.nan_to_num(np.asanyarray(img.dataobj, dtype=np.float32))
    if img.shape[:3] == ref.shape[:3] and np.allclose(img.affine, ref.affine):
        return d
    M = np.linalg.inv(img.affine) @ ref.affine
    g = np.indices(ref.shape[:3], dtype=np.float32).reshape(3, -1)
    g = np.vstack([g, np.ones((1, g.shape[1]), np.float32)])
    return map_coordinates(d, (M @ g)[:3], order=1, cval=0).reshape(ref.shape[:3])


def contrast_names(analysis_dir, n):
    """Name each contrast from the formula's term order and the contrast matrix."""
    try:
        formula = json.load(open(os.path.join(analysis_dir, "dataset_dict.json")))["neuroimaging_regression"]["formula"]
        terms = [t.strip() for t in formula.split("~")[1].split("+")]
    except Exception:
        terms = [t.strip() for t in os.path.basename(analysis_dir).split("-on-")[1].replace("-", "+").split("+")]
    terms = [re.sub(r"^(cluster|domain)_", "", t) for t in terms]
    try:
        C = np.load(os.path.join(analysis_dir, "contrast_matrix.npy"))
    except Exception:
        C = np.eye(len(terms))
    names = []
    for i in range(n):
        row = C[i] if i < len(C) else None
        if row is None or len(row) != len(terms):
            names.append(f"contrast {i}"); continue
        nz = np.flatnonzero(np.abs(row) > 1e-9)
        if len(nz) == 1:
            names.append(f"{terms[nz[0]]} vs 0")
        elif row.max() > 0 and np.allclose(row.sum(), 0):
            names.append(f"{terms[int(np.argmax(row))]} vs others")
        else:
            names.append(f"contrast {i} {np.round(row, 2).tolist()}")
    return names


def main(filters):
    ref = nib.load(MASK); mask = np.asanyarray(ref.dataobj) > 0
    sheet = pd.read_csv(SHEET)
    sheet = sheet[(sheet.get("selected", 1) != 0) & sheet[IMAGE_COLUMN].notna()].reset_index(drop=True)
    lab = pd.read_csv(os.path.join(SCA_ROOT, LABELS_FROM))
    domain = dict(zip(lab.symptom, lab.domain))
    symptoms = [s for d in DOMAIN_ORDER for s in lab.symptom if domain[s] == d]
    Y = sheet[symptoms].astype(float)
    paths = sheet[IMAGE_COLUMN]
    if BIDS_PREFIX: paths = paths.str.replace(BIDS_PREFIX[0], BIDS_PREFIX[1], regex=False)
    P = np.stack([resample(nib.load(p), ref)[mask] for p in paths])

    runs = sorted(r for r in glob.glob(os.path.join(SCA_ROOT, RUN_GLOB))
                  if any(k in r.lower() for k in RUN_KEYWORDS)
                  and (not filters or any(f in r for f in filters)))
    for reg in runs:
        maps = sorted(glob.glob(os.path.join(reg, "contrast_tval_FWE_*.nii*")),
                      key=lambda f: int(re.search(r"FWE_(\d+)\.nii", f).group(1)) if re.search(r"FWE_(\d+)\.nii", f) else 99)
        maps = [f for f in maps if re.search(r"FWE_\d+\.nii(\.gz)?$", f)]
        if not maps: continue
        analysis = os.path.dirname(reg)
        rel = os.path.relpath(analysis, SCA_ROOT).split(os.sep)
        tag = f"{rel[0]}__{rel[1]}"
        names = contrast_names(analysis, len(maps))
        # Three-line title (tree / analysis / contrast): one line is too long to fit a plot.
        X = pd.DataFrame({f"{rel[0]}\n{rel[1]}\n{nm}": P @ resample(nib.load(f), ref)[mask] for f, nm in zip(maps, names)})
        CORR = np.array([[spearmanr(X[c][Y[s].notna()], Y[s][Y[s].notna()])[0] for s in symptoms] for c in X.columns])
        out = os.path.join(OUT_DIR, tag); os.makedirs(out, exist_ok=True)
        pd.DataFrame(CORR, index=[c.replace("\n", " | ") for c in X.columns], columns=symptoms).to_csv(os.path.join(out, "rho.csv"))
        sa = SpecificityAnalyzer(X, Y.fillna(0), domain, correlation="spearman", out_dir=out, absval=False)
        sa._plot(CORR, sort_within_labels=False, smooth_by_label=True)
        print(f"{tag}: {len(maps)} maps", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
