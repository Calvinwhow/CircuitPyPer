import os
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image

from calvin_utils.neuroimaging_utils.nifti_utils.damage_score_utils import DamageScorer


MetricName = Literal[
    "spatial_correlation",
    "cosine",
    "sum",
    "avg_in_target",
    "avg_in_subject",
    "num_in_roi",
    "dice",
    "max_in_roi",
    "min_in_roi",
]


@dataclass
class ThresholdSpec:
    mode: Literal["above", "below"] = "above"
    value: float | None = None


class PredictionDamageScorer:
    """
    Orchestrates scalar scoring of voxelwise predictions against per-patient NIfTIs.

    - Uses the same patient order as the provided DataFrame.
    - Pairs row i with prediction_{i}.nii(.gz).
    - Optionally binarizes patient NIfTIs via a threshold.
    - Delegates metric computation to DamageScorer helpers (no reimplementation).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mask_path: str | None = None,
        prediction_prefix: str = "prediction_",
        prediction_exts: Iterable[str] = ("nii.gz", "nii"),
        resample_interpolation: str = "nearest",
        verbose: bool = False,
    ):
        self.df = df
        self.mask_path = mask_path
        self.prediction_prefix = prediction_prefix
        self.prediction_exts = tuple(prediction_exts)
        self.resample_interpolation = resample_interpolation
        self.verbose = verbose

        self._mask_img = nib.load(mask_path) if mask_path else None
        self._mask_data = self._mask_img.get_fdata() if self._mask_img else None
        self._mask_flat = self._mask_data.flatten() > 0 if self._mask_img else None

        self._scorer = DamageScorer(mask_path=mask_path)

    def add_score_column(
        self,
        patient_nifti_col: str,
        *,
        metric: MetricName = "avg_in_subject",
        threshold: ThresholdSpec | None = None,
        column_name: str | None = None,
        index_offset: int = 0,
        reference_nifti_path: str | None = None,
        prediction_dir: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute a scalar score per patient.

        Provide exactly one of:
        - reference_nifti_path: a single shared reference map for all patients
        - prediction_dir: paired prediction_{i}.nii(.gz) per row
        """
        if patient_nifti_col not in self.df.columns:
            raise ValueError(f"Column not found: {patient_nifti_col}")

        use_reference = reference_nifti_path is not None
        use_predictions = prediction_dir is not None
        if use_reference == use_predictions:
            raise ValueError("Provide exactly one of reference_nifti_path or prediction_dir.")

        if use_reference:
            ref_vec = self._load_as_vector(reference_nifti_path, threshold=None)

        scores: list[float] = []
        for i, path in enumerate(self.df[patient_nifti_col].tolist()):
            patient_vec = self._load_as_vector(path, threshold=threshold)
            if use_reference:
                ref_vec_i = ref_vec
            else:
                pred_path = self._find_prediction_path(prediction_dir, i + index_offset)
                if not pred_path:
                    raise FileNotFoundError(f"Missing prediction file for index {i + index_offset}")
                ref_vec_i = self._load_as_vector(pred_path, threshold=None)

            if patient_vec.shape != ref_vec_i.shape:
                raise ValueError(
                    f"Shape mismatch for row {i}: patient_vec {patient_vec.shape} vs reference {ref_vec_i.shape}"
                )

            score = self._compute_metric(metric, patient_vec, ref_vec_i)
            scores.append(float(score))

        out_col = column_name or f"{metric}_score"
        self.df[out_col] = np.array(scores, dtype=float)
        return self.df

    def _find_prediction_path(self, prediction_dir: str, idx: int) -> str | None:
        for ext in self.prediction_exts:
            candidate = os.path.join(prediction_dir, f"{self.prediction_prefix}{idx}.{ext}")
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_as_vector(self, path: str, threshold: ThresholdSpec | None) -> np.ndarray:
        if not isinstance(path, str) or not path:
            raise ValueError("Invalid NIfTI path")

        img = nib.load(path)
        if self._mask_img is not None:
            same_shape = img.shape == self._mask_img.shape
            same_affine = np.allclose(img.affine, self._mask_img.affine)
            if not (same_shape and same_affine):
                if self.verbose:
                    print(
                        f"[resample] {path} shape={img.shape}, "
                        f"zooms={img.header.get_zooms()[:3]} -> "
                        f"space shape={self._mask_img.shape}, "
                        f"zooms={self._mask_img.header.get_zooms()[:3]}"
                    )
                img = image.resample_to_img(
                    img,
                    self._mask_img,
                    interpolation=self.resample_interpolation,
                    force_resample=True,
                    copy_header=True,
                )

        data = img.get_fdata()
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if threshold is not None and threshold.value is not None:
            if threshold.mode == "above":
                data = (data > threshold.value).astype(np.float32)
            elif threshold.mode == "below":
                data = (data < threshold.value).astype(np.float32)
            else:
                raise ValueError(f"Unknown threshold mode: {threshold.mode}")

        flat = data.flatten()
        if self._mask_flat is not None:
            flat = flat[self._mask_flat]
        return flat

    def _compute_metric(self, metric: MetricName, subj: np.ndarray, roi: np.ndarray) -> float:
        if metric == "spatial_correlation":
            return self._scorer._calculate_spatial_correlation(subj, roi)
        if metric == "cosine":
            return self._scorer._calculate_cosine_similarity(subj, roi)
        if metric == "sum":
            return self._scorer._calculate_dot_product(subj, roi)
        if metric == "avg_in_target":
            return self._scorer._calculate_normalized_dot_product(subj, roi, denominator="avg_in_target")
        if metric == "avg_in_subject":
            return self._scorer._calculate_normalized_dot_product(subj, roi, denominator="avg_in_subject")
        if metric == "num_in_roi":
            return self._scorer._count_voxels_greater_than_threshold(subj, mask=roi, threshold=2)
        if metric == "dice":
            return self._scorer._calculate_dice(subj, roi)
        if metric == "max_in_roi":
            return self._scorer._calculate_max_in_roi(subj, roi)
        if metric == "min_in_roi":
            return self._scorer._calculate_min_in_roi(subj, roi)
        raise ValueError(f"Unsupported metric: {metric}")
