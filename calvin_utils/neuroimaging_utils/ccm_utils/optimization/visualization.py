"""Render optimization weight histories as GIFs in native map space."""

from pathlib import Path

import numpy as np

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.history import (
    OptimizationHistory,
)


def render_optimization_history(history, output_path=None, **kwargs):
    """Render an OptimizationHistory object or ``.npz`` path to a GIF.

    This is the reusable entry point for optimization pipelines and the CLI.
    When ``output_path`` is omitted for an archive, the GIF is written beside
    it as ``optimization.gif``.
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


def render_gif(history, output_path, *, fps=5, max_frames=60,
               view="auto", output_type=None, mask_path=None, vmax=None,
               max_lines=10000, viewer_url=None):
    """Create a weight/loss GIF, with anatomical slices when I/O supports them."""
    if fps <= 0 or max_frames <= 0:
        raise ValueError("fps and max_frames must be positive.")
    if history.weights.shape[0] == 0:
        raise ValueError("The history has no scored iterations.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    output_type = output_type or history.output_type
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
        from calvin_utils.neuroimaging_utils.output_functions import NeuroimageFileOutporter

        exporter = NeuroimageFileOutporter(output_ftype=output_type, mask_path=mask_path)
        exporter.validate_for_output()
        if spatial and not hasattr(exporter.io, "_map_to_image"):
            raise ValueError(f"No spatial image renderer is available for {output_type!r}.")

    n_iterations = history.weights.shape[0]
    indices = np.unique(np.linspace(
        0, n_iterations - 1, num=min(max_frames, n_iterations), dtype=int
    ))

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
    return output_path
