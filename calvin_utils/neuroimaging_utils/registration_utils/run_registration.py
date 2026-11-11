import os
import csv
from dataclasses import fields
from pathlib import Path

import numpy as np
import nibabel as nib

from calvin_utils.neuroimaging_utils.registration_utils.schema import RegistrationOutputs
from calvin_utils.neuroimaging_utils.registration_utils.skull_strip.skull_strip import SkullStrip


class RunRegistration:
    """
    Every scan in a CSV skull-stripped and registered to MNI, saved beside the scan, its output paths written back.

    Stages of run: read the CSV (IO); add the output columns (preprocess); run SkullStrip on each scan
    (process); save its brain, mask and transform beside the scan, BIDS-style, and note their paths in its
    row (postprocess); write the CSV back (IO).

    The CSV is held as one numpy array of strings, shape (1 + scans, columns), dtype object: row 0 is the
    header, each further row one scan. Python's csv module reads and writes its text, so quoted fields (a
    path holding a comma) survive.

    BIDS-style names, for sub-01_T1w.nii.gz and the rigid frame (affine: MNIaffine):
    sub-01_space-MNIrigid_desc-brain_T1w.nii.gz, sub-01_desc-brain_mask.nii.gz and
    sub-01_from-native_to-MNIrigid_xfm.txt; the entities go before the scan's suffix.
    """

    def __init__(self, input_csv, nifti_column="nifti_path", template_path=None, template_mask_path=None, affine=False, n_samples=50000, bins=32, seed=0):
        """
        input_csv (str): path to a CSV with one row per scan; the output paths are written back into it.
        nifti_column (str): name of the CSV column holding each scan's NIfTI path (.nii or .nii.gz).
        template_path (str or None): path to the MNI template (NIfTI file); None is the packaged MNI152_T1_2mm_brain.nii.
        template_mask_path (str or None): path to the template's brain mask (NIfTI file); None is the packaged MNI152_T1_2mm_brain_mask_filled.nii.
        affine (bool): True returns each brain in the full affine registration (stretched to MNI); False in its rigid simplification (own size and shape).
        n_samples (int): number of MNI brain points the registration's mutual information is scored at, at least 1000.
        bins (int): number of histogram bins per image for the mutual information.
        seed (int): seed for drawing those sample points, so a run is repeatable.
        """
        self.input_csv = input_csv
        self.nifti_column = nifti_column
        self.template_path = template_path
        self.template_mask_path = template_mask_path
        self.affine = affine
        self.n_samples = n_samples
        self.bins = bins
        self.seed = seed

    ###INTERNAL METHODS###
    def _read_scan_table(self):
        """The CSV as an array of strings, shape (1 + scans, columns), row 0 the header; short rows are padded with empty strings."""
        with open(self.input_csv, newline="") as handle:
            rows = [row for row in csv.reader(handle) if row]
        if len(rows) < 2:
            raise ValueError(f"{self.input_csv} needs a header row and at least one scan row")
        width = max(len(row) for row in rows)
        table = np.array([row + [""] * (width - len(row)) for row in rows], dtype=object)
        if self.nifti_column not in table[0]:
            raise ValueError(f"{self.input_csv} has no column {self.nifti_column!r}; its columns are {', '.join(table[0])}")
        missing = [path for path in table[1:, self._column(table, self.nifti_column)] if not Path(path).expanduser().is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} scan(s) in {self.nifti_column!r} do not exist, e.g. {missing[0]}")
        return table

    def _column(self, table, name):
        """The index of column ``name`` in the table's header row."""
        return int(np.flatnonzero(table[0] == name)[0])

    def _add_output_columns(self, table):
        """The table with a column for each output path (header = the RegistrationOutputs field name), kept when already present."""
        for field in fields(RegistrationOutputs):
            if field.name not in table[0]:
                column = np.full((len(table), 1), "", dtype=object)
                column[0, 0] = field.name
                table = np.hstack([table, column])
        return table

    def _skull_strip(self, nifti_path):
        """A SkullStrip of one scan, with this run's template and settings."""
        return SkullStrip(str(Path(nifti_path).expanduser()), self.template_path, self.template_mask_path, affine=self.affine, n_samples=self.n_samples, bins=self.bins, seed=self.seed)

    def _output_paths(self, nifti_path):
        """The BIDS-style output paths beside one scan."""
        nifti_path = Path(nifti_path).expanduser()
        stem = nifti_path.name.removesuffix(".gz").removesuffix(".nii")
        prefix = stem.rpartition("_")[0] if "_" in stem else stem
        suffix = stem.rpartition("_")[2] if "_" in stem else ""
        frame = "MNIaffine" if self.affine else "MNIrigid"
        brain_name = f"{prefix}_space-{frame}_desc-brain_{suffix}.nii.gz" if suffix else f"{prefix}_space-{frame}_desc-brain.nii.gz"
        folder = nifti_path.parent
        return RegistrationOutputs(folder / brain_name, folder / f"{prefix}_desc-brain_mask.nii.gz", folder / f"{prefix}_from-native_to-{frame}_xfm.txt")

    def _save_outputs(self, nifti_path, result, outputs):
        """Write one scan's brain (its header, the frame's affine), mask (its own space) and transform to its output paths."""
        scan = nib.load(str(Path(nifti_path).expanduser()))
        brain_img = nib.Nifti1Image(result.brain, result.brain_affine, header=scan.header)
        brain_img.set_sform(result.brain_affine, code=int(scan.header["sform_code"]) or 1)
        # A qform holds rotation, shift and voxel size only: it takes the rigid frame, and is cleared for the affine one.
        brain_img.set_qform(None if self.affine else result.brain_affine, code=0 if self.affine else (int(scan.header["qform_code"]) or 1))
        nib.save(brain_img, outputs.brain_path)
        nib.save(nib.Nifti1Image(result.brain_mask.astype(np.uint8), scan.affine), outputs.brain_mask_path)
        np.savetxt(outputs.transform_path, result.transform, fmt="%.10f")

    def _record_outputs(self, table, row, outputs):
        """Put one scan's output paths in its row of the table."""
        for field in fields(RegistrationOutputs):
            table[row, self._column(table, field.name)] = str(getattr(outputs, field.name))

    def _write_scan_table(self, table):
        """Write the table back over input_csv."""
        with open(self.input_csv, "w", newline="") as handle:
            csv.writer(handle).writerows(table.tolist())

    ###PROPERTY METHODS###
    @property
    def input_csv(self):
        return self._input_csv

    @input_csv.setter
    def input_csv(self, path):
        path = str(Path(str(path)).expanduser())
        if not os.path.isfile(path):
            raise ValueError(f"File does not exist: {path}")
        self._input_csv = path

    @property
    def nifti_column(self):
        return self._nifti_column

    @nifti_column.setter
    def nifti_column(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError(f"nifti_column must be a column name, got {name!r}")
        self._nifti_column = name

    @property
    def template_path(self):
        return self._template_path

    @template_path.setter
    def template_path(self, path):
        if path is not None and not os.path.isfile(str(path)):
            raise ValueError(f"File does not exist: {path}")
        self._template_path = None if path is None else str(path)

    @property
    def template_mask_path(self):
        return self._template_mask_path

    @template_mask_path.setter
    def template_mask_path(self, path):
        if path is not None and not os.path.isfile(str(path)):
            raise ValueError(f"File does not exist: {path}")
        self._template_mask_path = None if path is None else str(path)

    @property
    def affine(self):
        return self._affine

    @affine.setter
    def affine(self, value):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"affine must be True or False, got {value!r}")
        self._affine = bool(value)

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        self._n_samples = int(value)

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, value):
        self._bins = int(value)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = int(value)

    ###BUSINESS LOGIC###
    def run(self):
        """
        Returns (np.ndarray, shape (scans, 3), dtype object): each scan's output paths as strings, columns in RegistrationOutputs field order; also written back to input_csv.
        """
        table = self._read_scan_table()
        table = self._add_output_columns(table)
        nifti_index = self._column(table, self.nifti_column)
        for row in range(1, len(table)):
            nifti_path = table[row, nifti_index]
            print(f"registering {nifti_path}", flush=True)
            result = self._skull_strip(nifti_path).run()
            outputs = self._output_paths(nifti_path)
            self._save_outputs(nifti_path, result, outputs)
            self._record_outputs(table, row, outputs)
        self._write_scan_table(table)
        output_indices = [self._column(table, field.name) for field in fields(RegistrationOutputs)]
        return table[1:, output_indices]
