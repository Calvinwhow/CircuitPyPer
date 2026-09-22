#!/usr/bin/env python3
"""
Symptom-specificity bars for optimized convergent maps and comparison maps,
each scored in its own modality's native data.

Every map is tested against the patient data it was built from: network maps
against connectivity_t_path, vlsm maps against Nifti_File_Path, and fiber maps
against the native fiber_path_guerrera values (never voxelized). Each patient's
score is the cosine similarity of their native vector with the map (the score the
optimizer maximizes). The score is Spearman-correlated with every symptom and the
rhos are plotted grouped motor | cognitive | emotional in one fixed symptom order.

One folder per map set and modality: an SVG per map plus rho.csv.
This is not a CLI. Change the values in the CONFIG section, then run.
Optional argv: substrings; only map sets whose name contains one of them are run.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

SCRIPTS_DIR = Path(__file__).resolve().parent
CIRCUIT_PYPER_DIR = SCRIPTS_DIR.parent
for p in (str(CIRCUIT_PYPER_DIR.parent), str(CIRCUIT_PYPER_DIR)):
    if p not in sys.path: sys.path.insert(0, p)
from calvin_utils.neuroimaging_utils.ccm_utils.symptom_specificity import SpecificityAnalyzer
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.oof_pipeline import load_patient_images
from calvin_utils.file_utils.import_functions import GiiNiiFileImport

# =============================================================================
# CONFIG
# =============================================================================
SCA_ROOT = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/symptom_on_lhs"
SHEET = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/optimized_master_list_filtered_HigherIsWorse.csv"
BIDS_PREFIX = None           # (old, new) to rewrite image paths when the drive is mounted elsewhere
VOLUME_MASK = str(CIRCUIT_PYPER_DIR / "resources" / "MNI152_T1_2mm_brain_mask.nii")
FIBER_ATLAS = "/Volumes/OneTouch/resources/Atlas_tck_MNI/Atlas_all30_MNI.npz"
# modality: (patient image column, mask/atlas, output type)
NATIVE = {
    "network": ("connectivity_t_path", VOLUME_MASK, "nii"),
    "vlsm": ("Nifti_File_Path", VOLUME_MASK, "nii"),
    "fiber": ("fiber_path_guerrera", FIBER_ATLAS, "fiber"),
}
LABELS_FROM = "vlsm_regressions_clusters/cluster_regression_input.csv"   # symptom -> domain
OPT_ROOT = "optimization_HigherIsBetter_3cluster_insample"
CLUSTERS = ["motor", "cognitive", "emotional"]
OPT_MAP = {"network": "optimized_map.nii.gz", "vlsm": "optimized_map.nii.gz", "fiber": "optimized_map.fib.npy"}
TOTALS_TREE = "{m}_regressions_HigherIsWorse-TestTotalScores"
TOTALS = ["TotalBarsScore", "TotalCCASRawScore", "CNRSTotScore"]
TOTAL_MAP = {"network": "contrast_tval_0.nii.gz", "vlsm": "contrast_tval_0.nii.gz", "fiber": "contrast_tval_0.fib.values.npy"}
# Original 3-cluster regressions on the ranked HigherIsBetter symptom maps (identity contrasts).
ORIG_CLUSTER_DIR = {
    "network": "network_regressions_clusters/cluster_regression_identity_standardized/Nifti_File_Path-on-cluster_motor-cluster_cognitive-cluster_emotional",
    "vlsm": "vlsm_regressions_clusters/cluster_regression_identity_standardized/Nifti_File_Path-on-cluster_motor-cluster_cognitive-cluster_emotional",
    "fiber": "fiber_regressions_clusters/cluster_regression_identity_standardized/Fiber_File_Path-on-cluster_motor-cluster_cognitive-cluster_emotional",
}
ORIG_EXT = {"network": ".nii.gz", "vlsm": ".nii.gz", "fiber": ".fib.npy"}
# Weighted regression of every pool map on the cluster indicators, weights = signed optimized weights.
OPTW_REG = "regressions-HigherIsBetter-optWeights3cluster"
OPTW_DIR = "regression_identity_and_vs_others/Image_File_Path-on-domain_motor + domain_cognitive + domain_emotional"
OPTW_CONTRASTS = ["motor vs 0", "cognitive vs 0", "emotional vs 0",
                  "motor vs others", "cognitive vs others", "emotional vs others"]
# Original VLSM cluster regressions (cluster_k ~ image_path over the symptom maps), FWE maps.
PROPER_CLUSTERS = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/proper_regressions_clusters"
PROPER_DOMAINS = ["motor", "emotional", "cognitive"]        # cluster 0, 1, 2
# {set name: {modality: {plot label: map path relative to SCA_ROOT, or absolute}}}
MAP_SETS = {
    OPT_ROOT: {m: {f"{c}\noptimized map": f"{OPT_ROOT}/{m}_{c}/{OPT_MAP[m]}" for c in CLUSTERS}
               for m in NATIVE},
    "original_clusters_ranked_HigherIsBetter": {
        m: {f"{c}\noriginal cluster map": f"{ORIG_CLUSTER_DIR[m]}/regression/contrast_tval_{i}{ORIG_EXT[m]}"
            for i, c in enumerate(CLUSTERS)} for m in NATIVE},
    OPTW_REG: {
        m: {f"{name}\noptWeights regression": f"{m}_{OPTW_REG}/{OPTW_DIR}/regression/contrast_tval_{i}{ORIG_EXT[m]}"
            for i, name in enumerate(OPTW_CONTRASTS)} for m in NATIVE},
    "proper_regressions_clusters_FWE": {
        "vlsm": {f"cluster {i} ({d})\ncontrast_tval_FWE_0": f"{PROPER_CLUSTERS}/cluster_{i}-on-image_path/regression/contrast_tval_FWE_0.nii.gz"
                 for i, d in enumerate(PROPER_DOMAINS)}},
    "total_scores_native": {m: {f"{t}\ncontrast_tval_0": f"{TOTALS_TREE.format(m=m)}/{t}-on-{NATIVE[m][0]}/regression/{TOTAL_MAP[m]}"
                                for t in TOTALS} for m in NATIVE},
}
OUT_ROOT = os.path.join(SCA_ROOT, "specificity_native")
DOMAIN_ORDER = ["motor", "cognitive", "emotional"]
# =============================================================================


def load_maps(paths, mask):
    """Maps in the patients' feature order. FWE maps are NaN where nothing
    survived; those locations are set to 0 so they contribute nothing."""
    importer = GiiNiiFileImport(import_path=None, mask_path=mask)
    return np.nan_to_num(np.asarray(importer._import_matrices(list(paths)).T, dtype=np.float32))


def main(filters):
    sheet = pd.read_csv(SHEET)
    sheet = sheet[sheet.get("selected", 1) != 0].reset_index(drop=True)
    lab = pd.read_csv(os.path.join(SCA_ROOT, LABELS_FROM))
    domain = dict(zip(lab.symptom, lab.domain))
    symptoms = [s for d in DOMAIN_ORDER for s in lab.symptom if domain[s] == d]
    sets = {k: v for k, v in MAP_SETS.items() if not filters or any(f in k for f in filters)}

    for modality, (image_col, mask, output_type) in NATIVE.items():
        rows = sheet[sheet[image_col].notna()].reset_index(drop=True)
        if BIDS_PREFIX:
            rows[image_col] = rows[image_col].astype(str).str.replace(*BIDS_PREFIX, regex=False)
        P = load_patient_images(rows, image_col, mask, output_type=output_type)
        P_norm = np.linalg.norm(P, axis=1)
        Y = rows[symptoms].astype(float)
        for set_name, by_modality in sets.items():
            maps = by_modality.get(modality, {})
            present = {label: os.path.join(SCA_ROOT, rel) for label, rel in maps.items()
                       if os.path.isfile(os.path.join(SCA_ROOT, rel))}
            missing = [rel for rel in maps.values() if not os.path.isfile(os.path.join(SCA_ROOT, rel))]
            if missing:
                print(f"{set_name}/{modality}: missing {missing}", flush=True)
            if not present:
                continue
            M = load_maps(present.values(), mask)
            if M.shape[1] != P.shape[1]:
                raise ValueError(f"{set_name}/{modality}: maps have {M.shape[1]} features, patients {P.shape[1]}.")
            cosine = (P @ M.T) / (P_norm[:, None] * np.linalg.norm(M, axis=1)[None, :])
            X = pd.DataFrame(cosine, columns=[f"{modality}\n{label}" for label in present])
            CORR = np.array([[spearmanr(X[c][Y[s].notna()], Y[s][Y[s].notna()])[0] for s in symptoms]
                             for c in X.columns])
            out = os.path.join(OUT_ROOT, set_name, modality); os.makedirs(out, exist_ok=True)
            pd.DataFrame(CORR, index=[c.replace("\n", " | ") for c in X.columns],
                         columns=symptoms).to_csv(os.path.join(out, "rho.csv"))
            sa = SpecificityAnalyzer(X, Y.fillna(0), domain, correlation="spearman", out_dir=out, absval=False)
            sa._plot(CORR, sort_within_labels=False, smooth_by_label=True)
            print(f"{set_name}/{modality}: {len(present)} maps, {len(rows)} patients ({image_col}) -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
