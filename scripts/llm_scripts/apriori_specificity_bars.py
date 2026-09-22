#!/usr/bin/env python3
"""
Symptom-specificity bars for the a priori maps.

For every map in each a priori folder: score each patient as the overlap of their
Nifti_File_Path image with the map (sum of patient x map over the brain mask),
Spearman-correlate that score with every symptom in the sheet, and plot the rhos
grouped motor | cognitive | emotional -- same symptoms, same order, every plot.

This is not a CLI. Change the values in the CONFIG section, then run.
"""
import os, sys, glob, re
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy.stats import spearmanr

CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
for p in (str(CIRCUIT_PYPER_DIR.parent), str(CIRCUIT_PYPER_DIR)):
    if p not in sys.path: sys.path.insert(0, p)
from calvin_utils.neuroimaging_utils.ccm_utils.symptom_specificity import SpecificityAnalyzer

# =============================================================================
# CONFIG
# =============================================================================
SCA_ROOT = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/symptom_on_lhs"
SHEET = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/optimized_master_list_filtered_HigherIsWorse.csv"
IMAGE_COLUMN = "Nifti_File_Path"
MASK = str(CIRCUIT_PYPER_DIR / "resources" / "MNI152_T1_2mm_brain_mask.nii")
RUNS = [                                  # folder under SCA_ROOT holding the a priori regression
    "vlsm_regressions_HigherIsWorse-Ranked-aPriori",
    "vlsm_regressions-HigherIsBetter-aPriori",
    "network_regressions_HigherIsWorse-Ranked-aPriori",
    "network_regressions-HigherIsBetter-aPriori",
]
MAP_PATTERN = "contrast_tval_FWE_{i}.nii.gz"   # FWE: only significant voxels score
CONTRASTS = {0: "motor vs 0", 1: "cognitive vs 0", 2: "emotional vs 0",
             3: "motor vs others", 4: "cognitive vs others", 5: "emotional vs others"}
OUT_DIR = os.path.join(SCA_ROOT, "specificity_aPriori")
DOMAIN_ORDER = ["motor", "cognitive", "emotional"]
# =============================================================================


def main():
    mask = np.asanyarray(nib.load(MASK).dataobj) > 0
    sheet = pd.read_csv(SHEET)
    sheet = sheet[(sheet.get("selected", 1) != 0) & sheet[IMAGE_COLUMN].notna()].reset_index(drop=True)

    # Symptom order: taken once from the first run's input, grouped by domain.
    inp = pd.read_csv(glob.glob(os.path.join(SCA_ROOT, RUNS[0], "regression_input.csv"))[0])
    domain = dict(zip(inp.symptom, inp.domain))
    symptoms = [s for d in DOMAIN_ORDER for s in inp.symptom if domain[s] == d]
    Y = sheet[symptoms].astype(float)

    print(f"loading {len(sheet)} patient images ...", flush=True)
    P = np.stack([np.nan_to_num(np.asanyarray(nib.load(p).dataobj, dtype=np.float32))[mask]
                  for p in sheet[IMAGE_COLUMN]])            # (patients, voxels)

    for run in RUNS:
        reg = glob.glob(os.path.join(SCA_ROOT, run, "regression_identity_and_vs_others", "*", "regression"))
        if not reg:
            print(f"skip {run}: not run yet"); continue
        scores = {}
        for i, name in CONTRASTS.items():
            f = os.path.join(reg[0], MAP_PATTERN.format(i=i))
            if not os.path.exists(f): continue
            m = np.nan_to_num(np.asanyarray(nib.load(f).dataobj, dtype=np.float32))[mask]
            scores[f"{run} | {name}"] = P @ m
        X = pd.DataFrame(scores)
        # Spearman on pairwise-complete patients: cognitive has fewer rows than motor.
        CORR = np.zeros((X.shape[1], len(symptoms)))
        for a, xc in enumerate(X.columns):
            for b, s in enumerate(symptoms):
                ok = Y[s].notna().to_numpy()
                CORR[a, b] = spearmanr(X[xc][ok], Y[s][ok])[0]
        pd.DataFrame(CORR, index=X.columns, columns=symptoms).to_csv(
            os.path.join(OUT_DIR, f"{run}_rho.csv") if os.makedirs(OUT_DIR, exist_ok=True) is None else None)
        sa = SpecificityAnalyzer(X, Y.fillna(0), domain, correlation="spearman",
                                 out_dir=os.path.join(OUT_DIR, run), absval=False)
        sa._plot(CORR, sort_within_labels=False, smooth_by_label=True)
        print(f"{run}: {X.shape[1]} plots -> {os.path.join(OUT_DIR, run)}", flush=True)


if __name__ == "__main__":
    main()
