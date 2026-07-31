import os
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from calvin_utils.file_utils.import_functions import GiiNiiFileImport
from calvin_utils.neuroimaging_utils.output_functions import NeuroimageFileOutporter
from calvin_utils.plotting_utils.colormaps import resolve_cmap


PathLike = Union[str, os.PathLike]


class SUITCerebellumPlotter:
    """
    Import NIfTI data with Calvin's existing loader and plot it on a SUITPy
    cerebellar flatmap.

    SUITPy expects a NIfTI file path for ``vol_to_surf``. For an existing NIfTI,
    this class can plot the original file directly. When data has been loaded
    through ``GiiNiiFileImport`` and transformed in memory, ``export_loaded_map``
    writes a NIfTI with ``NeuroimageFileOutporter`` before plotting.
    """

    def __init__(
        self,
        import_path: Optional[Union[PathLike, pd.Series]] = None,
        file_pattern: Optional[str] = None,
        file_column: Optional[str] = None,
        mask_path: Optional[PathLike] = "default",
        out_dir: Optional[PathLike] = None,
        process_special_values: bool = True,
    ):
        self.import_path = import_path
        self.file_pattern = file_pattern
        self.file_column = file_column
        self.mask_path = mask_path
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.process_special_values = process_special_values

        self.importer = None
        self.data = None
        self.file_paths = []
        self.surface_data = None

    @staticmethod
    def _require_suitpy():
        try:
            import SUITPy as suit
        except ImportError as exc:
            raise ImportError(
                "SUITPy is required for cerebellum plotting. Install it in the "
                "active environment with `pip install SUITPy`."
            ) from exc
        return suit

    @staticmethod
    def _as_single_file_import(path: PathLike):
        path = Path(path)
        if path.is_file():
            return str(path.parent), path.name
        return str(path), None

    @staticmethod
    def _strip_nifti_suffix(path_or_name: PathLike) -> str:
        stem = Path(str(path_or_name)).name
        for suffix in (".nii.gz", ".nii"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return Path(stem).stem

    def load(self):
        """
        Load the requested NIfTI(s) with ``GiiNiiFileImport``.

        Returns
        -------
        pandas.DataFrame
            Imported matrix with one column per input file.
        """
        import_path = self.import_path
        file_pattern = self.file_pattern

        if isinstance(import_path, (str, os.PathLike)) and file_pattern is None:
            import_path, file_pattern = self._as_single_file_import(import_path)

        self.importer = GiiNiiFileImport(
            import_path=import_path,
            file_column=self.file_column,
            file_pattern=file_pattern,
            process_special_values=self.process_special_values,
            mask_path=self.mask_path,
        )
        self.data = self.importer.run()
        self.file_paths = getattr(self.importer, "file_paths", [])
        return self.data

    def _select_loaded_map(self, column: Optional[Union[str, int]] = None):
        if self.data is None:
            self.load()

        if self.data.shape[1] == 1 and column is None:
            return self.data.iloc[:, 0], self.data.columns[0]
        if column is None:
            raise ValueError(
                "Multiple maps are loaded. Provide `column` as a column name or integer index."
            )
        if isinstance(column, int):
            return self.data.iloc[:, column], self.data.columns[column]
        return self.data.loc[:, column], column

    def export_loaded_map(
        self,
        column: Optional[Union[str, int]] = None,
        out_dir: Optional[PathLike] = None,
        file_name: Optional[str] = None,
    ) -> str:
        """
        Save one loaded map back to NIfTI using Calvin's existing output helper.
        """
        map_data, column_name = self._select_loaded_map(column)
        output_dir = Path(out_dir) if out_dir is not None else self.out_dir
        if output_dir is None:
            output_dir = Path.cwd() / "suit_cerebellum_plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        if file_name is None:
            stem = self._strip_nifti_suffix(column_name)
            file_name = f"{stem}_for_suit.nii.gz"
        file_stem = file_name
        for suffix in (".nii.gz", ".nii"):
            if file_stem.endswith(suffix):
                file_stem = file_stem[: -len(suffix)]

        outporter = NeuroimageFileOutporter(output_ftype="nii", mask_path=self.mask_path)
        outporter.save_map(map_data.to_numpy(), file_name=file_stem, out_dir=str(output_dir))
        return str(output_dir / f"{file_stem}.nii.gz")

    def vol_to_surface(
        self,
        nifti_path: Optional[PathLike] = None,
        column: Optional[Union[str, int]] = None,
        space: str = "SUIT",
        **vol_to_surf_kwargs: Any,
    ):
        """
        Project a NIfTI volume to the SUITPy cerebellar surface.

        Use ``space='SUIT'`` for data already in SUIT space. Use another SUITPy
        supported space, such as ``'MNI'``, when appropriate for the input image.
        """
        suit = self._require_suitpy()
        if nifti_path is None:
            if (
                column is None
                and isinstance(self.import_path, (str, os.PathLike))
                and Path(self.import_path).is_file()
            ):
                nifti_path = self.import_path
            elif column is None and len(self.file_paths) == 1:
                nifti_path = self.file_paths[0]
            else:
                nifti_path = self.export_loaded_map(column=column)

        self.surface_data = suit.vol_to_surf(str(nifti_path), space=space, **vol_to_surf_kwargs)
        return self.surface_data

    def plot_flatmap(
        self,
        data=None,
        nifti_path: Optional[PathLike] = None,
        column: Optional[Union[str, int]] = None,
        space: str = "SUIT",
        out_file: Optional[PathLike] = None,
        cmap: str = "bwr",
        cscale: Optional[list] = None,
        threshold: Optional[float] = None,
        colorbar: bool = True,
        render: str = "matplotlib",
        new_figure: bool = True,
        dpi: int = 300,
        **plot_kwargs: Any,
    ):
        """
        Plot data on the SUITPy flatmap and optionally save the figure.
        """
        suit = self._require_suitpy()
        cmap = resolve_cmap(cmap)
        if data is None:
            data = self.vol_to_surface(nifti_path=nifti_path, column=column, space=space)
        data = np.asarray(data).squeeze()
        if "underlay" in plot_kwargs and plot_kwargs["underlay"] is not None:
            plot_kwargs["underlay"] = np.asarray(plot_kwargs["underlay"]).squeeze()

        ax = suit.flatmap.plot(
            data=data,
            cmap=cmap,
            cscale=cscale,
            threshold=threshold,
            colorbar=colorbar,
            render=render,
            new_figure=new_figure,
            **plot_kwargs,
        )

        if out_file is not None:
            out_file = Path(out_file)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
        return ax

    def run(
        self,
        nifti_path: Optional[PathLike] = None,
        column: Optional[Union[str, int]] = None,
        space: str = "SUIT",
        out_file: Optional[PathLike] = None,
        **plot_kwargs: Any,
    ):
        """
        Convenience wrapper: project a NIfTI to surface and plot the flatmap.
        threshold (scalar or 2-element array)
            Threshold for functional overlay. If one value is given, only values above are shown
            If two values are given, values below lower threshold or above upper threshold are shown
        """
        return self.plot_flatmap(
            nifti_path=nifti_path,
            column=column,
            space=space,
            out_file=out_file,
            **plot_kwargs,
        )

    def plot_without_background(
        self,
        nifti_path: Optional[PathLike] = None,
        column: Optional[Union[str, int]] = None,
        space: str = "SUIT",
        out_file: Optional[PathLike] = None,
        **plot_kwargs: Any,
    ):
        """
        Plot a cerebellar flatmap with only the thresholded overlay visible.
        """
        data = self.vol_to_surface(nifti_path=nifti_path, column=column, space=space)
        plot_kwargs.setdefault("underlay", np.zeros_like(np.asarray(data).squeeze()))
        plot_kwargs.setdefault("undermap", ListedColormap(["white", "white"]))
        plot_kwargs.setdefault("underscale", [0, 1])
        plot_kwargs.setdefault("borders", None)
        plot_kwargs.setdefault("backgroundcolor", "white")
        return self.plot_flatmap(
            data=data,
            out_file=out_file,
            **plot_kwargs,
        )

    def isolate_cerebellum(self, t1_path: PathLike, **isolate_kwargs: Any):
        """
        Run SUITPy cerebellum isolation for an anatomical T1 image.
        """
        suit = self._require_suitpy()
        return suit.isolate(str(t1_path), **isolate_kwargs)

    def normalize_to_suit(
        self,
        source_file: PathLike,
        mask_file: PathLike,
        **normalize_kwargs: Any,
    ):
        """
        Run SUITPy anatomical normalization to SUIT space.
        """
        suit = self._require_suitpy()
        return suit.normalize(
            source_file=str(source_file),
            mask_file=str(mask_file),
            **normalize_kwargs,
        )

    def reslice_to_suit(
        self,
        source_image: PathLike,
        deformation: PathLike,
        mask: PathLike,
        out_file: Optional[PathLike] = None,
        voxelsize: int = 2,
        **reslice_kwargs: Any,
    ):
        """
        Apply a SUITPy deformation to a functional/statistical image.
        """
        import nibabel as nib

        suit = self._require_suitpy()
        img = suit.reslice_image(
            source_image=str(source_image),
            deformation=str(deformation),
            mask=str(mask),
            voxelsize=voxelsize,
            **reslice_kwargs,
        )
        if out_file is not None:
            out_file = Path(out_file)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            nib.save(img, str(out_file))
        return img
