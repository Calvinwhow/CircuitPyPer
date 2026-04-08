import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from nilearn import image
from scipy.signal import butter, sosfiltfilt, coherence
from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import VolumetricTimeSeriesIO


class TimeseriesConnectivity:
    """
    Derive a seed-based volumetric connectivity map from a patient ROI and a
    normative/reference 4D volumetric time series.

    Purpose
    -------
    This class takes a 3D patient ROI, extracts the average time series within
    that ROI from a normative 4D volumetric dataset, and computes voxelwise
    connectivity between that seed time series and every other voxel in the
    masked brain. Connectivity is computed separately within many overlapping
    frequency bands, producing a 4D NIfTI output where the first three
    dimensions are space and the fourth dimension is frequency-band index.

    This output is intended to be readable by the same volumetric-timeseries
    import pipeline used elsewhere in the codebase. In practice, this means the
    saved 4D NIfTI can later be imported, flattened across space x band.

    Conceptual model
    ----------------
    The workflow is:

    1. Load a normative/reference 4D time series in common space.
       This may represent source-localized MEG, electrophysiology, rs-fMRI, or
       any other volumetric time series already aligned into a common space.

    2. Load a patient ROI in the same space, or resample it to the reference
       time series grid if needed.

    3. Extract the average time series from all voxels in the ROI.

    4. Create a sliding frequency filterbank spanning a user-defined range.

    5. For each band, bandpass both the seed time series and every voxel time
       series, then compute a connectivity metric between the seed and each
       voxel.

    6. Save the resulting voxelwise connectivity maps across bands as a 4D
       NIfTI, where dim4 = band index.

    Interpretation
    --------------
    The saved 4D connectivity NIfTI is not a raw electrophysiologic time series.
    Its fourth dimension represents frequency bands, not time samples.

    If connectivity_metric == "correlation", each 3D volume represents the
    band-limited Pearson correlation between the ROI seed signal and every
    voxel.

    If connectivity_metric == "coherence", each 3D volume represents
    frequency-specific linear synchrony between the ROI seed signal and every
    voxel, summarized within each band.

    If connectivity_metric == "mutual_information", each 3D volume represents
    statistical dependence between the ROI seed signal and every voxel within
    each band, without restricting that dependence to be linear.

    Parameters
    ----------
    reference_timeseries_path : str
        Path to the normative/reference 4D NIfTI. This must already be in the
        target volumetric space used for analysis. The data are expected to have
        shape (x, y, z, t), where t is the number of time samples.

    mask_path : str, default='default'
        Path to a 3D mask NIfTI defining the spatial analysis domain. The mask
        only constrains spatial voxels, not the fourth dimension. If set to
        'default', a predefined MNI volumetric mask is used. If set to None, no
        spatial masking is applied and the full reference volume is used.

    roi_threshold : float, default=0.0
        Threshold applied to the patient ROI after loading/resampling. Voxels
        with values greater than this threshold are considered part of the ROI.
        This allows use of binary ROIs or probabilistic/weighted ROIs that must
        be binarized.

    mask_threshold : float, default=0.0
        Threshold applied to the spatial analysis mask. Voxels with values
        greater than this threshold are included in the analysis domain.

    tr : float or None, default=None
        Sampling interval in seconds. This is used to infer the sampling
        frequency fs = 1 / tr for frequency filtering and coherence analysis.
        If None, the value is inferred from the fourth zoom in the NIfTI
        header. For example, MEG sampled at 1000 Hz should use tr=0.001.

    connectivity_metric : {"correlation", "coherence", "mutual_information"}, default="correlation"
        Connectivity metric used after bandpass filtering.

        "correlation":
            Uses band-limited Pearson correlation. This is the simplest and most
            stable first-pass measure. It captures similarity of fluctuations
            within each band but does not explicitly model phase synchrony.

        "coherence":
            Uses magnitude-squared coherence. This is more specifically an
            oscillatory coupling measure and is often more interpretable if the
            biological question concerns synchronized rhythmic activity.

        "mutual_information":
            Uses a mutual-information estimator within each band. This is more
            general and can detect nonlinear dependence, but is harder to
            interpret mechanistically and can be more estimator-sensitive.

    fisher_z : bool, default=False
        If True and connectivity_metric == "correlation", apply a Fisher
        z-transform to correlation values before saving. This is often useful if
        the resulting maps will be treated as approximately normal continuous
        variables in later analyses. This does not apply to coherence or mutual
        information.

    Notes on frequency parameters
    -----------------------------
    This class uses a sliding filterbank rather than a few canonical broad bands
    such as delta/theta/alpha/beta/gamma. The goal is to make the fourth
    dimension behave like a smooth spectral axis that can later be visualized or
    regressed across continuously.

    The following parameters are passed later when generating connectivity:

    freq_min : float
        The lower bound of the frequency range to analyze.

    freq_max : float
        The upper bound of the frequency range to analyze.

    n_bands : int
        Number of bands to generate.

    bandwidth : float or None
        Width of each band in Hz. If None, it is derived automatically from
        freq_min, freq_max, n_bands, and overlap.

    overlap : float
        Fractional overlap between adjacent bands. Must satisfy
        0 <= overlap < 1. Larger values produce smoother progression across the
        spectral axis but also increase redundancy between adjacent bands.

    filter_order : int
        Butterworth filter order. Higher values produce sharper filters but can
        increase ringing or instability in some situations.

    Practical parameter choices
    ---------------------------
    For source-localized MEG sampled at 1000 Hz, the main adjustment is usually
    the frequency range rather than the class structure itself. For example,
    tr=0.001, freq_min around 1 Hz, and freq_max somewhere in the low- to
    mid-gamma range are typically more sensible than fMRI-like defaults such as
    0.01-0.25 Hz.

    Outputs
    -------
    The main output is a 4D NIfTI with shape (x, y, z, n_bands). Each 3D volume
    corresponds to one sliding band. This file is intended for both
    visualization and downstream regression input.

    The class can also return the masked voxel × band array internally when
    needed, but the primary persistent object should be the 4D NIfTI.

    Important assumptions
    ---------------------
    The interpretation of the result depends entirely on what the reference 4D
    time series represents. If the reference dataset is a true voxelwise or
    source-localized volumetric time series in common space, then the output is
    a volumetric connectivity map. If the reference file is parcelwise data
    projected into voxels, the code still runs, but the biological meaning is
    different.

    This class assumes the ROI is 3D and the reference dataset is 4D. The ROI is
    treated as a seed region only. The result is always a seed-to-whole-brain
    connectivity object.

    This class does not infer causal direction. It derives a normative
    connectivity profile from a seed defined by the patient ROI.
    """
    def __init__(
        self,
        reference_timeseries_path,
        mask_path='default',
        roi_threshold=0.0,
        mask_threshold=0.0,
        tr=0.001,       # sampling rate (0.0001->1000hz)
        connectivity_metric='coherence',
        fisher_z=True,
    ):
        self.reference_timeseries_path = reference_timeseries_path
        self.mask_path = mask_path
        self.roi_threshold = float(roi_threshold)
        self.mask_threshold = float(mask_threshold)
        self.connectivity_metric = connectivity_metric
        self.fisher_z = fisher_z

        self.reference_img = nib.load(reference_timeseries_path)
        self.reference_data = self.reference_img.get_fdata(dtype=np.float32)

        if self.reference_data.ndim != 4:
            raise ValueError(
                f"reference_timeseries_path must be 4D NIfTI, got shape {self.reference_data.shape}"
            )

        self.reference_shape = self.reference_data.shape[:3]
        self.n_timepoints = self.reference_data.shape[3]
        self.reference_affine = self.reference_img.affine

        self.tr = float(tr) if tr is not None else self._infer_tr(self.reference_img)
        self.fs = 1.0 / self.tr

        self.timeseries_io = VolumetricTimeSeriesIO(
            mask_path=mask_path,
            threshold=mask_threshold,
        )
        self.mask_path_resolved = self.timeseries_io.resolved_mask_path

        self._mask_vec = self._resolve_mask_vector()
        self._masked_reference_ts = None

    @staticmethod
    def _safe_stem(path):
        name = os.path.basename(path)
        if name.endswith(".nii.gz"):
            return name[:-7]
        return os.path.splitext(name)[0]

    @staticmethod
    def _infer_tr(img):
        zooms = img.header.get_zooms()
        if len(zooms) < 4:
            raise ValueError("Could not infer TR from header; provide tr explicitly.")
        tr = float(zooms[3])
        if tr <= 0:
            raise ValueError("Invalid TR in header; provide tr explicitly.")
        return tr

    def _resolve_mask_vector(self):
        if self.mask_path_resolved is None:
            return np.ones(np.prod(self.reference_shape), dtype=bool)

        mask_img = nib.load(self.mask_path_resolved)
        mask_data = mask_img.get_fdata(dtype=np.float32)

        if mask_data.shape != self.reference_shape or not np.allclose(mask_img.affine, self.reference_affine):
            mask_img = image.resample_to_img(
                mask_img,
                image.index_img(self.reference_img, 0),
                interpolation="nearest",
            )
            mask_data = mask_img.get_fdata(dtype=np.float32)

        return mask_data.flatten() > self.mask_threshold

    def _get_masked_reference_timeseries(self):
        """
        Returns:
            (n_masked_voxels, n_timepoints)
        """
        if self._masked_reference_ts is None:
            flat = self.reference_data.reshape(-1, self.n_timepoints)
            self._masked_reference_ts = flat[self._mask_vec, :]
        return self._masked_reference_ts

    def _load_patient_roi_mask(self, roi_path):
        roi_img = nib.load(roi_path)

        if roi_img.ndim != 3:
            raise ValueError(f"ROI must be 3D NIfTI, got shape {roi_img.shape} in {roi_path}")

        if roi_img.shape != self.reference_shape or not np.allclose(roi_img.affine, self.reference_affine):
            roi_img = image.resample_to_img(
                roi_img,
                image.index_img(self.reference_img, 0),
                interpolation="nearest",
            )

        roi_data = roi_img.get_fdata(dtype=np.float32)
        roi_mask = roi_data > self.roi_threshold

        if not np.any(roi_mask):
            raise ValueError(f"ROI contains no voxels above threshold in {roi_path}")

        return roi_mask

    def _extract_seed_timeseries(self, roi_path):
        roi_mask = self._load_patient_roi_mask(roi_path)
        seed_ts = self.reference_data[roi_mask, :].mean(axis=0)
        return np.asarray(seed_ts, dtype=np.float32)

    @staticmethod
    def _make_band_edges(freq_min, freq_max, n_bands, bandwidth=None, overlap=0.5):
        if n_bands < 1:
            raise ValueError("n_bands must be >= 1")
        if not (0 <= overlap < 1):
            raise ValueError("overlap must satisfy 0 <= overlap < 1")
        if freq_max <= freq_min:
            raise ValueError("freq_max must be > freq_min")

        if bandwidth is None:
            if n_bands == 1:
                return [(freq_min, freq_max)]
            step = (freq_max - freq_min) / n_bands
            bandwidth = step * (1.0 + overlap)

        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0")

        step = bandwidth * (1.0 - overlap)
        if step <= 0:
            raise ValueError("Band step must be > 0")

        bands = []
        lo = freq_min
        while lo < freq_max and len(bands) < n_bands:
            hi = min(lo + bandwidth, freq_max)
            if hi > lo:
                bands.append((float(lo), float(hi)))
            lo += step

        if len(bands) == 0:
            raise ValueError("No bands were generated.")

        return bands

    def _bandpass(self, arr, f_lo, f_hi, order=4):
        nyq = 0.5 * self.fs
        if f_lo <= 0 or f_hi >= nyq or f_lo >= f_hi:
            raise ValueError(
                f"Invalid band ({f_lo}, {f_hi}) for fs={self.fs} Hz; nyquist={nyq} Hz"
            )

        sos = butter(order, [f_lo / nyq, f_hi / nyq], btype='bandpass', output='sos')
        return sosfiltfilt(sos, arr, axis=-1)

    @staticmethod
    def _rowwise_correlation(vox_by_time, seed_ts, eps=1e-8):
        seed = seed_ts - seed_ts.mean()
        seed_sd = np.sqrt(np.mean(seed ** 2)) + eps

        vox = vox_by_time - vox_by_time.mean(axis=1, keepdims=True)
        vox_sd = np.sqrt(np.mean(vox ** 2, axis=1)) + eps

        numer = np.mean(vox * seed[None, :], axis=1)
        corr = numer / (vox_sd * seed_sd)

        corr = np.clip(corr, -0.999999, 0.999999)
        return corr.astype(np.float32)

    @staticmethod
    def _rowwise_correlation(vox_by_time, seed_ts, eps=1e-8):
        """
        Pearson correlation between each voxel timeseries and the seed timeseries.

        Parameters
        ----------
        vox_by_time : array, shape (n_voxels, n_time)
        seed_ts     : array, shape (n_time,)

        Returns
        -------
        corr : array, shape (n_voxels,)
        """
        seed = seed_ts - seed_ts.mean()
        seed_sd = np.sqrt(np.mean(seed ** 2)) + eps

        vox = vox_by_time - vox_by_time.mean(axis=1, keepdims=True)
        vox_sd = np.sqrt(np.mean(vox ** 2, axis=1)) + eps

        numer = np.mean(vox * seed[None, :], axis=1)
        corr = numer / (vox_sd * seed_sd)
        corr = np.clip(corr, -0.999999, 0.999999)
        return corr.astype(np.float32)


    def _rowwise_coherence(self, vox_by_time, seed_ts, nperseg=None, noverlap=None):
        """
        Mean magnitude-squared coherence between each voxel timeseries and the seed timeseries.

        This is intended to be used after bandpass filtering, so it averages coherence
        over the remaining in-band frequencies.

        Parameters
        ----------
        vox_by_time : array, shape (n_voxels, n_time)
        seed_ts     : array, shape (n_time,)
        nperseg     : int or None
        noverlap    : int or None

        Returns
        -------
        coh : array, shape (n_voxels,)
        """
        n_vox, n_time = vox_by_time.shape

        if nperseg is None:
            nperseg = min(256, n_time)
        if noverlap is None:
            noverlap = nperseg // 2

        out = np.zeros(n_vox, dtype=np.float32)

        for i in range(n_vox):
            _, cxy = coherence(
                seed_ts,
                vox_by_time[i, :],
                fs=self.fs,
                nperseg=nperseg,
                noverlap=noverlap,
            )
            out[i] = np.mean(cxy, dtype=np.float64)

        return out


    @staticmethod
    def _rowwise_mutual_information(vox_by_time, seed_ts, n_bins=16, eps=1e-12):
        """
        Histogram-based mutual information between each voxel timeseries and the seed timeseries.

        Parameters
        ----------
        vox_by_time : array, shape (n_voxels, n_time)
        seed_ts     : array, shape (n_time,)
        n_bins      : int

        Returns
        -------
        mi : array, shape (n_voxels,)
        """
        seed_ts = np.asarray(seed_ts, dtype=np.float64)
        vox_by_time = np.asarray(vox_by_time, dtype=np.float64)

        seed_edges = np.histogram_bin_edges(seed_ts, bins=n_bins)
        seed_digit = np.clip(np.digitize(seed_ts, seed_edges[1:-1], right=False), 0, n_bins - 1)

        out = np.zeros(vox_by_time.shape[0], dtype=np.float32)

        for i in range(vox_by_time.shape[0]):
            x = vox_by_time[i, :]
            x_edges = np.histogram_bin_edges(x, bins=n_bins)
            x_digit = np.clip(np.digitize(x, x_edges[1:-1], right=False), 0, n_bins - 1)

            joint = np.zeros((n_bins, n_bins), dtype=np.float64)
            np.add.at(joint, (x_digit, seed_digit), 1.0)
            joint /= joint.sum()

            px = joint.sum(axis=1, keepdims=True)
            py = joint.sum(axis=0, keepdims=True)

            valid = joint > 0
            mi = np.sum(joint[valid] * np.log((joint[valid] + eps) / ((px @ py)[valid] + eps)))
            out[i] = mi

        return out

    def _compute_band_connectivity(
        self,
        vox_by_time,
        seed_ts,
        metric='coherence',
        coherence_nperseg=None,
        coherence_noverlap=None,
        mi_n_bins=16,
    ):
        if metric == 'correlation':
            conn = self._rowwise_correlation(vox_by_time, seed_ts)
            if self.fisher_z:
                conn = np.arctanh(np.clip(conn, -0.999999, 0.999999)).astype(np.float32)
            return conn

        if metric == 'coherence':
            return self._rowwise_coherence(
                vox_by_time,
                seed_ts,
                nperseg=coherence_nperseg,
                noverlap=coherence_noverlap,
            )

        if metric == 'mutual_information':
            return self._rowwise_mutual_information(
                vox_by_time,
                seed_ts,
                n_bins=mi_n_bins,
            )

        raise NotImplementedError(f"Unsupported connectivity_metric: {metric}")
    def generate_connectivity_array(
        self,
        roi_path,
        freq_min=1,
        freq_max=150,
        n_bands=100,
        bandwidth=4,
        overlap=0.5,
        filter_order=4,
    ):
        """
        Generate and save one 4D connectivity NIfTI per ROI.

        Parameters
        ----------
        roi_paths : list[str]
            List of patient ROI NIfTI paths. Each ROI is treated as a seed region
            and will produce one separate connectivity output file.

        out_dir : str
            Output directory where the derived connectivity NIfTIs will be written.

        freq_min : float, default=0.01
            Lower bound of the analyzed frequency range in Hz.
            (rsfMRI default: 0.01 | MEG default: 1.0)

        freq_max : float, default=0.25
            Upper bound of the analyzed frequency range in Hz.
            (rsfMRI default: 0.25 | MEG default: 150)

        n_bands : int, default=100
            Number of sliding frequency bands to generate between freq_min and
            freq_max.

        bandwidth : float or None, default=None
            Width of each band in Hz. If None, the width is derived automatically so
            that approximately n_bands overlapping bands span the requested range.

        overlap : float, default=0.5
            Fractional overlap between neighboring bands. Must satisfy
            0 <= overlap < 1.

        filter_order : int, default=4
            Butterworth filter order used for bandpass filtering.
            (rsfMRI default: None | MEG default: 4)

        file_suffix : str, default=".tsconn"
            Suffix inserted before ".nii.gz" in the saved output filename.

        save_band_info : bool, default=True
            If True, save a JSON sidecar describing the band edges and key analysis
            settings used to derive each 4D NIfTI.

        Output
        ------
        For each ROI, this method saves a 4D NIfTI with shape (x, y, z, n_bands),
        where each 3D volume is the voxelwise connectivity map for one sliding
        frequency band.

        Returns
        -------
        list[str]
            Paths to the saved 4D NIfTI files.
        """
        seed_ts = self._extract_seed_timeseries(roi_path)
        masked_reference = self._get_masked_reference_timeseries()

        bands = self._make_band_edges(
            freq_min=freq_min,
            freq_max=freq_max,
            n_bands=n_bands,
            bandwidth=bandwidth,
            overlap=overlap,
        )

        conn_masked = np.zeros((masked_reference.shape[0], len(bands)), dtype=np.float32)

        for b_idx, (f_lo, f_hi) in enumerate(bands):
            seed_f = self._bandpass(seed_ts, f_lo, f_hi, order=filter_order)
            vox_f = self._bandpass(masked_reference, f_lo, f_hi, order=filter_order)
            conn_masked[:, b_idx] = self._compute_band_connectivity(
                vox_by_time=vox_f,
                seed_ts=seed_f,
                metric=self.connectivity_metric,
            )

        return conn_masked, bands

    def generate_connectivity_vector(
        self,
        roi_path,
        freq_min=0.01,
        freq_max=0.25,
        n_bands=100,
        bandwidth=None,
        overlap=0.5,
        filter_order=4,
    ):
        conn_masked, bands = self.generate_connectivity_array(
            roi_path=roi_path,
            freq_min=freq_min,
            freq_max=freq_max,
            n_bands=n_bands,
            bandwidth=bandwidth,
            overlap=overlap,
            filter_order=filter_order,
        )
        return conn_masked.reshape(-1).astype(np.float32), bands

    def save_connectivity_nifti(
        self,
        roi_path,
        out_path,
        freq_min=0.01,
        freq_max=0.25,
        n_bands=100,
        bandwidth=None,
        overlap=0.5,
        filter_order=4,
        save_band_info=True,
    ):
        """
        Saves connectivity as a 4D NIfTI where dim4 = band index.

        The write path uses VolumetricTimeSeriesIO.save_files(...), so the
        output is compatible with your existing nifti-timeseries import path.
        """
        conn_masked, bands = self.generate_connectivity_array(
            roi_path=roi_path,
            freq_min=freq_min,
            freq_max=freq_max,
            n_bands=n_bands,
            bandwidth=bandwidth,
            overlap=overlap,
            filter_order=filter_order,
        )

        fake_target = out_path
        if fake_target.endswith(".nii.gz"):
            fake_target = fake_target[:-7]
        elif fake_target.endswith(".nii"):
            fake_target = fake_target[:-4]

        self.timeseries_io.save_files(
            arr=conn_masked,
            file_paths=[fake_target],
            dry_run=False,
            file_suffix="",
        )

        saved_nii = f"{fake_target}.nii.gz"

        if save_band_info:
            sidecar_path = f"{fake_target}.bands.json"
            payload = {
                "reference_timeseries_path": self.reference_timeseries_path,
                "mask_path": self.mask_path_resolved,
                "tr": self.tr,
                "connectivity_metric": self.connectivity_metric,
                "fisher_z": self.fisher_z,
                "bands_hz": [[float(lo), float(hi)] for lo, hi in bands],
            }
            with open(sidecar_path, "w") as f:
                json.dump(payload, f, indent=2)

        return saved_nii

    def generate_and_save_from_paths(
        self,
        roi_paths,
        out_dir,
        freq_min=0.01,
        freq_max=0.25,
        n_bands=100,
        bandwidth=None,
        overlap=0.5,
        filter_order=4,
        file_suffix=".tsconn",
        save_band_info=True,
    ):
        os.makedirs(out_dir, exist_ok=True)

        saved_paths = []
        for roi_path in tqdm(roi_paths, desc="Saving timeseries connectivity nifti files"):
            stem = self._safe_stem(roi_path)
            out_path = os.path.join(out_dir, f"{stem}{file_suffix}.nii.gz")
            saved = self.save_connectivity_nifti(
                roi_path=roi_path,
                out_path=out_path,
                freq_min=freq_min,
                freq_max=freq_max,
                n_bands=n_bands,
                bandwidth=bandwidth,
                overlap=overlap,
                filter_order=filter_order,
                save_band_info=save_band_info,
            )
            saved_paths.append(saved)

        return saved_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate 4D volumetric connectivity NIfTIs from patient ROIs and a normative/reference 4D timeseries."
    )

    parser.add_argument("--reference_timeseries", required=True, help="Path to normative/reference 4D NIfTI")
    parser.add_argument("--rois", nargs="+", required=True, help="List of patient ROI NIfTIs")
    parser.add_argument("--out_dir", required=True, help="Output directory for connectivity NIfTIs")

    parser.add_argument("--mask", default="default", help="Optional 3D spatial mask")
    parser.add_argument("--tr", type=float, default=None, help="TR in seconds; if omitted, inferred from header")
    parser.add_argument("--roi_threshold", type=float, default=0.0, help="Threshold for ROI binarization")
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="Threshold for spatial mask")

    parser.add_argument("--freq_min", type=float, default=0.01, help="Minimum frequency")
    parser.add_argument("--freq_max", type=float, default=0.25, help="Maximum frequency")
    parser.add_argument("--n_bands", type=int, default=100, help="Number of sliding bands")
    parser.add_argument("--bandwidth", type=float, default=None, help="Bandwidth in Hz")
    parser.add_argument("--overlap", type=float, default=0.5, help="Band overlap fraction")
    parser.add_argument("--filter_order", type=int, default=4, help="Butterworth filter order")

    parser.add_argument("--metric", default="correlation", choices=["correlation"], help="Connectivity metric")
    parser.add_argument("--fisher_z", action="store_true", help="Apply Fisher z-transform to correlations")
    parser.add_argument("--no_band_info", action="store_true", help="Do not save band sidecar JSON")

    args = parser.parse_args()

    mapper = TimeseriesConnectivity(
        reference_timeseries_path=args.reference_timeseries,
        mask_path=args.mask,
        roi_threshold=args.roi_threshold,
        mask_threshold=args.mask_threshold,
        tr=args.tr,
        connectivity_metric=args.metric,
        fisher_z=args.fisher_z,
    )

    mapper.generate_and_save_from_paths(
        roi_paths=args.rois,
        out_dir=args.out_dir,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        n_bands=args.n_bands,
        bandwidth=args.bandwidth,
        overlap=args.overlap,
        filter_order=args.filter_order,
        save_band_info=not args.no_band_info,
    )