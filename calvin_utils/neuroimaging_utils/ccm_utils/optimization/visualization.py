"""Render and export optimization histories in native map space."""

from pathlib import Path
import tempfile
import warnings

import numpy as np

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.history import (
    OptimizationHistory,
)


def render_optimization_history(history, output_path=None, **kwargs):
    """Render an OptimizationHistory object or ``.npz`` path.

    This is the reusable entry point for optimization pipelines and the CLI.
    The GIF and a native map stack are written together. When ``output_path``
    is omitted for an archive, the GIF is written beside it as
    ``optimization.gif``.
    """
    history_path = None
    if not isinstance(history, OptimizationHistory):
        history_path = Path(history).expanduser().resolve()
        history = OptimizationHistory.load(history_path)
    if output_path is None:
        if history_path is None:
            raise ValueError("output_path is required for an in-memory history.")
        output_path = history_path.with_name("optimization.gif")
    return render_gif(history, output_path, **kwargs)


def _spatial_image(values, exporter):
    """Ask generic image I/O to put a feature vector in anatomical space."""
    import nibabel as nib

    image = exporter.io._map_to_image(values)
    volume = np.asarray(nib.as_closest_canonical(image).dataobj)
    if volume.ndim != 3:
        raise ValueError(f"Spatial visualization needs a 3D image, got {volume.shape}.")
    return volume


def _slices(volume, cuts):
    return (
        volume[cuts[0], :, :].T,
        volume[:, cuts[1], :].T,
        volume[:, :, cuts[2]].T,
    )


def _normalized_output_type(output_type):
    return {
        "nifti": "nii",
        "gii": "surface",
        "freesurfer": "surface",
    }.get(output_type, output_type)


def optimization_stack_path(gif_path, output_type):
    """Return the default native map-stack path paired with a GIF."""
    output_type = _normalized_output_type(output_type)
    gif_path = Path(gif_path)
    if output_type == "nii":
        return gif_path.with_suffix(".nii.gz")
    if output_type == "fiber":
        return gif_path.with_suffix(".fib.npy")
    return None


def _frame_indices(n_iterations, max_frames):
    """Evenly sample a history while always retaining its endpoints."""
    if max_frames is None:
        return np.arange(n_iterations, dtype=int)
    if max_frames <= 0:
        raise ValueError("max_frames must be positive or None.")
    return np.unique(np.linspace(
        0, n_iterations - 1,
        num=min(max_frames, n_iterations), dtype=int,
    ))


def export_optimization_map_stack(history, output_path, *, output_type=None,
                                  mask_path=None, max_frames=60):
    """Reconstruct sampled weight snapshots as one native map stack.

    Snapshots are sampled evenly across the full history, including the first
    and last. Pass ``max_frames=None`` to retain every stored snapshot. NIfTI
    histories are saved as ``(x, y, z, snapshot)``. Fiber histories are saved
    as ``(fiber, snapshot)`` and are restored to full atlas order when
    optimization used a masked subset of the atlas.
    """
    if history.weights.shape[0] == 0:
        raise ValueError("The history has no scored iterations.")

    output_type = _normalized_output_type(output_type or history.output_type)
    mask_path = mask_path or history.mask_path or None
    if output_type not in {"nii", "fiber"}:
        raise ValueError(
            "Optimization map stacks are supported for NIfTI and fiber "
            f"histories, not {output_type!r}."
        )

    from calvin_utils.neuroimaging_utils.io.exporters import NeuroimageFileOutporter

    exporter = NeuroimageFileOutporter(
        output_ftype=output_type, mask_path=mask_path
    )
    exporter.validate_for_output()
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = _frame_indices(history.weights.shape[0], max_frames)
    n_snapshots = len(indices)

    if output_type == "nii":
        import nibabel as nib

        lower_name = output_path.name.lower()
        if not lower_name.endswith((".nii", ".nii.gz")):
            raise ValueError("A volumetric optimization stack must end in .nii or .nii.gz.")
        first_image = exporter.io._map_to_image(history.map_at(indices[0]))
        first_volume = np.asarray(first_image.dataobj, dtype=np.float32)
        if first_volume.ndim != 3:
            raise ValueError(
                "A volumetric optimization snapshot must be 3D, got "
                f"{first_volume.shape}."
            )
        temporary = tempfile.NamedTemporaryFile(
            prefix=".optimization_stack_", suffix=".dat",
            dir=output_path.parent, delete=False,
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        stacked = None
        try:
            stacked = np.memmap(
                temporary_path, mode="w+", dtype=np.float32,
                shape=first_volume.shape + (n_snapshots,),
            )
            stacked[..., 0] = first_volume
            for snapshot, iteration in enumerate(indices[1:], start=1):
                image = exporter.io._map_to_image(history.map_at(iteration))
                volume = np.asarray(image.dataobj, dtype=np.float32)
                if volume.shape != first_volume.shape:
                    raise ValueError(
                        "Optimization snapshots do not share a volume shape: "
                        f"{first_volume.shape} and {volume.shape}."
                    )
                stacked[..., snapshot] = volume
            stacked.flush()
            header = first_image.header.copy()
            header.set_data_dtype(np.float32)
            nib.save(
                nib.Nifti1Image(stacked, first_image.affine, header=header),
                str(output_path),
            )
        finally:
            if stacked is not None:
                del stacked
            temporary_path.unlink(missing_ok=True)
    else:
        if output_path.suffix.lower() != ".npy":
            raise ValueError("A fiber optimization stack must end in .npy.")
        fiber_mask = exporter.io.fiber_mask
        n_fibers = len(exporter.io.reference_fibers)
        n_features = history.maps.shape[1]
        if n_features not in {int(fiber_mask.sum()), n_fibers}:
            raise ValueError(
                "Optimization map length does not match the masked or full "
                f"fiber atlas: {n_features} versus {int(fiber_mask.sum())} "
                f"masked / {n_fibers} full fibers."
            )
        full_stack = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32,
            shape=(n_fibers, n_snapshots),
        )
        full_stack[:] = 0
        for snapshot, iteration in enumerate(indices):
            values = np.asarray(history.map_at(iteration), dtype=np.float32)
            if n_features == n_fibers:
                full_stack[:, snapshot] = values
            else:
                full_stack[fiber_mask, snapshot] = values
        full_stack.flush()
        del full_stack

    return output_path


def render_gif(history, output_path, *, fps=5, max_frames=60,
               view="auto", output_type=None, mask_path=None, vmax=None,
               max_lines=10000, viewer_url=None, write_map_stack=True,
               map_stack_path=None):
    """Create a weight/loss GIF and its volumetric or fiber map stack."""
    if fps <= 0 or max_frames <= 0:
        raise ValueError("fps and max_frames must be positive.")
    if history.weights.shape[0] == 0:
        raise ValueError("The history has no scored iterations.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    output_type = _normalized_output_type(output_type or history.output_type)
    mask_path = mask_path or history.mask_path or None
    if view == "auto":
        view = "circuit" if output_type == "fiber" else (
            "spatial" if output_type in {"nii", "nifti", "nii_timeseries"}
            else "metrics"
        )
    spatial = view == "spatial"
    circuit = view == "circuit"
    if circuit and output_type != "fiber":
        raise ValueError("Circuit Viewer fiber rendering requires a fiber history.")
    exporter = None
    if spatial or circuit:
        from calvin_utils.neuroimaging_utils.io.exporters import NeuroimageFileOutporter

        exporter = NeuroimageFileOutporter(output_ftype=output_type, mask_path=mask_path)
        exporter.validate_for_output()
        if spatial and not hasattr(exporter.io, "_map_to_image"):
            raise ValueError(f"No spatial image renderer is available for {output_type!r}.")

    n_iterations = history.weights.shape[0]
    indices = _frame_indices(n_iterations, max_frames)

    circuit_renderer = None
    if spatial or circuit:
        fig = plt.figure(figsize=(12, 6.5), layout="constrained")
        grid = fig.add_gridspec(2, 3, height_ratios=(2, 1))
        weight_ax = fig.add_subplot(grid[1, 0])
        loss_ax = fig.add_subplot(grid[1, 1:])
        if spatial:
            image_axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
            final_volume = _spatial_image(history.map_at(indices[-1]), exporter)
            nonzero = np.argwhere(np.isfinite(final_volume) & (final_volume != 0))
            cuts = tuple(np.median(nonzero, axis=0).astype(int)) if len(nonzero) else tuple(
                size // 2 for size in final_volume.shape
            )
            scale = float(vmax) if vmax is not None else float(
                np.percentile(np.abs(final_volume[np.isfinite(final_volume)]), 99)
            )
            if scale <= 0:
                scale = 1.0
            images = []
            for ax, values, title in zip(
                image_axes, _slices(final_volume, cuts), ("Sagittal", "Coronal", "Axial")
            ):
                images.append(ax.imshow(values, origin="lower", cmap="coolwarm",
                                        vmin=-scale, vmax=scale))
                ax.set_title(title)
                ax.axis("off")
            fig.colorbar(images[-1], ax=image_axes, shrink=0.68, label="Map value")
        else:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            from calvin_utils.neuroimaging_utils.ccm_utils.optimization.circuit_frames import CircuitFiberFrames

            scale = float(vmax) if vmax is not None else max(
                float(np.max(np.abs(history.map_at(index)))) for index in indices
            )
            scale = scale if scale > 0 else 1.0
            circuit_renderer = CircuitFiberFrames(
                mask_path, exporter.io, vmax=scale, max_lines=max_lines,
                viewer_url=viewer_url,
            )
            image_ax = fig.add_subplot(grid[0, :])
            circuit_image = image_ax.imshow(
                np.full((560, 900, 3), 255, dtype=np.uint8)
            )
            image_ax.set_title("Fiber map")
            image_ax.axis("off")
            fig.colorbar(
                ScalarMappable(norm=Normalize(-scale, scale), cmap="coolwarm"),
                ax=image_ax, shrink=0.7, label="Fiber value",
            )
            images = None
            cuts = None
    else:
        fig, (weight_ax, loss_ax) = plt.subplots(1, 2, figsize=(10, 4),
                                               layout="constrained")
        images = None
        cuts = None

    names = history.map_names if len(history.map_names) <= 12 else tuple(
        str(index + 1) for index in range(len(history.map_names))
    )
    positions = np.arange(len(names))
    bars = weight_ax.bar(positions, np.zeros(len(names)), color="#247ba0")
    weight_ax.set_xticks(positions, names, rotation=45, ha="right")
    weight_ax.set_ylabel("Weight")
    weight_ax.set_title("Component weights")
    limit = max(float(np.max(np.abs(history.weights))), 0.1) * 1.1
    weight_ax.set_ylim(-limit, limit)
    weight_ax.axhline(0, color="0.3", linewidth=0.8)

    steps = np.arange(1, n_iterations + 1)
    loss_ax.plot(steps, history.losses, color="0.75", linewidth=1.3)
    completed, = loss_ax.plot([], [], color="#247ba0", linewidth=2)
    marker, = loss_ax.plot([], [], "o", color="#e76f51")
    loss_ax.set_xlim(1, max(n_iterations, 2))
    lower, upper = float(np.min(history.losses)), float(np.max(history.losses))
    margin = max((upper - lower) * 0.1, 0.02)
    loss_ax.set_ylim(lower - margin, upper + margin)
    loss_ax.set_xlabel("Iteration")
    loss_ax.set_ylabel(history.objective_label)
    loss_ax.set_title("Optimization score")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    try:
        with writer.saving(fig, str(output_path), dpi=110):
            for index in indices:
                weights = history.weights[index]
                for bar, weight in zip(bars, weights):
                    bar.set_height(weight)
                    bar.set_color("#247ba0" if weight >= 0 else "#e76f51")
                completed.set_data(steps[:index + 1], history.losses[:index + 1])
                marker.set_data([index + 1], [history.losses[index]])
                if spatial:
                    volume = _spatial_image(history.map_at(index), exporter)
                    for image, values in zip(images, _slices(volume, cuts)):
                        image.set_data(values)
                elif circuit:
                    circuit_image.set_data(
                        circuit_renderer.render(history.map_at(index), index)
                    )
                fig.suptitle(f"Iteration {index + 1} / {n_iterations}  |  "
                             f"{history.objective_label}: {history.losses[index]:.3f}")
                writer.grab_frame()
    finally:
        if circuit_renderer is not None:
            circuit_renderer.close()
        plt.close(fig)
    if write_map_stack:
        stack_path = (
            Path(map_stack_path).expanduser()
            if map_stack_path is not None
            else optimization_stack_path(output_path, output_type)
        )
        if stack_path is None:
            warnings.warn(
                "No native optimization map stack was written for output type "
                f"{output_type!r}; only NIfTI and fiber histories are supported.",
                stacklevel=2,
            )
        else:
            export_optimization_map_stack(
                history, stack_path, output_type=output_type,
                mask_path=mask_path, max_frames=max_frames,
            )
            print(f"Saved optimization map stack to: {stack_path}")
    return output_path
